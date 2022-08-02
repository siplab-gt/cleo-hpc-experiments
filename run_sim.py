import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from brian2 import Network, mm, ms, StateMonitor, prefs

import cleosim
from cleosim.electrodes import Probe, TKLFPSignal
from cleosim import opto

from aussel_model.model import single_process3 as sp3
from aussel_model.interface import user_interface_simple as uis
from plot_results import plot_input, plot_lfp


def main(args):
    setup_start = time.time()
    prefs.codegen.target = args.target
    (net, all_ngs, elec_pos), params = setup_aussel_net(args)
    path = params[23]
    all_ngs_exc = [area[0][0] for area in all_ngs]
    all_ngs_inh = [area[1][0] for area in all_ngs]

    assign_coords(all_ngs)
    sim = cleosim.CLSimulator(net)

    config_processor(args, sim)

    lfp = TKLFPSignal("lfp", save_history=True)
    # use same electrode coordinates, with the same 150um scale
    probe = Probe("probe", elec_pos * 0.15 * mm, [lfp])
    for ng_exc, ng_inh in zip(all_ngs_exc, all_ngs_inh):
        orntn = orntn_for_ng(ng_exc)
        mean_orntn = np.mean(orntn, axis=0, keepdims=True)
        # inh groups don't have orientation information stored, so we will approximate
        # with the average orientation for the exc neurons in each region
        sim.inject_recorder(probe, ng_exc, tklfp_type="exc", orientation=orntn)
        sim.inject_recorder(probe, ng_inh, tklfp_type="inh", orientation=mean_orntn)
    light_params = opto.default_blue
    light_params["R0"] = args.R0 * mm  # bigger fiber radius
    op_int = opto.OptogeneticIntervention(
        "opto",
        opto.FourStateModel(opto.ChR2_four_state),
        light_params,
        (2.5, -7, 7.5) * mm,
        (-1, 1, 0),
        save_history=True,
        max_Irr0_mW_per_mm2=args.maxIrr0,
    )
    # sim.inject_stimulator(op_int, *all_ngs_exc, Iopto_var_name="Iopto")
    sim.inject_stimulator(op_int, all_ngs_exc[0], Iopto_var_name="Iopto")
    # mon_Iopto = StateMonitor(all_ngs_exc[0], 'Iopto', record=True)
    # sim.network.add(mon_Iopto)
    plot_viz(args, all_ngs_exc, all_ngs_inh, probe, op_int)

    print(f"Setup time: {(time.time()-setup_start)} seconds")

    if args.runtime > 0:
        sp3.run_process(net, all_ngs, elec_pos, *params)

        # fig, ax = plt.subplots()
        # ax.plot(mon_Iopto.t, mon_Iopto.Iopto.T)

        save_lfp(path, lfp)
        plot_lfp(path)
        save_input(path, op_int)
        plot_input(path)

    uis.aborted = False
    uis.save_plots()
    if args.show_plots:
        plt.show()


def config_processor(args, sim):
    dt_ms = 1
    t_start_ms = 0
    t_stop_ms = 300

    if args.mode == "orig":
        # this is equivalent to the RecordOnlyProcessor
        my_process = lambda state, t_ms: ({}, t_ms)
    elif args.mode == "OL":

        def my_process(state, t_ms):
            if t_start_ms <= t_ms < t_stop_ms:
                opto_val = args.Irr0_OL
            else:
                opto_val = 0
            return {"opto": opto_val}, t_ms

    elif args.mode == "fit":

        def my_process(state, t_ms):
            # one side of a normal distribution. Max is 3 st devs away
            opto_val = np.abs(args.maxIrr0 / 3 * np.random.randn())
            return {"opto": opto_val}, t_ms

    elif args.mode == "CL":
        pass

    # need to subclass so it's concrete
    class MyLIOP(cleosim.processing.LatencyIOProcessor):
        def process(self, state, t_ms):
            return my_process(state, t_ms)

    proc = MyLIOP(dt_ms)
    sim.set_io_processor(proc)


def plot_viz(args, all_ngs_exc, all_ngs_inh, probe, op_int):
    colors_exc = ["#fb9a99", "#fdbf6f", "#b2df8a", "#cab2d6"]
    colors_inh = ["#e31a1c", "#ff7f00", "#33a02c", "#6a3d9a"]
    colors = colors_exc + colors_inh
    fig, ax = cleosim.visualization.plot(
        *all_ngs_exc,
        *all_ngs_inh,
        zlim=(6.5, 9),
        colors=colors,
        invert_z=False,
        devices=[(probe, {"size": 5, "color": (0.1, 0.1, 0.1, 0.5), "marker": "."}), (op_int, {'n_points': 1e5})],
        scatterargs={"alpha": 0.8, "marker": ".", "s": 2 * 10000 / args.maxN},
        figsize=(3, 4),
    )
    ax.set(zticks=[7, 8, 9])
    ax.view_init(60, -75)
    ax.get_legend().remove()


def assign_coords(all_ngs):
    # all_ngs is 4 x 2 x 1 nested list.
    # 4 areas, 2 types, and I don't know why the last level:
    # aussel_model/model_files/preparation.py:337
    for i_area in range(4):  # EC, DG, CA3, CA1
        for i_type in range(2):  # exc, inh
            ng = all_ngs[i_area][i_type][0]
            cleosim.coordinates.assign_coords(
                ng, ng.x_soma / mm, ng.y_soma / mm, ng.z_soma / mm
            )


def orntn_for_ng(ng):
    xyz_dendrite = np.column_stack(
        (ng.x_dendrite / mm, ng.y_dendrite / mm, ng.z_dendrite / mm)
    )
    xyz_soma = np.column_stack((ng.x_soma / mm, ng.y_soma / mm, ng.z_soma / mm))
    assert xyz_dendrite.shape == xyz_soma.shape == (ng.N, 3)
    return xyz_dendrite - xyz_soma


def save_lfp(path, lfp: TKLFPSignal):
    # imitate method from Aussel 2018: take average signal from one cylinder
    # of contacts and subtract from the other
    lfp1 = lfp.lfp_uV[:, :144].mean(axis=1)
    lfp2 = lfp.lfp_uV[:, 144:288].mean(axis=1)
    fname = os.path.join(path, "tklfp.npy")
    np.save(fname, lfp2 - lfp1)


def save_input(path, op_int: opto.OptogeneticIntervention):
    fname = os.path.join(path, "input.npz")
    npzfile = np.load(fname)
    Irr0_mW_per_mm2 = np.array(op_int.values)
    np.savez_compressed(
        fname, Irr0_mW_per_mm2=Irr0_mW_per_mm2, t_opto_ms=op_int.t_ms, **npzfile
    )


def setup_aussel_net(args) -> Network:
    uis.maxN.set(args.maxN)
    uis.runtime.set(args.runtime)
    uis.f1.set(args.f1)
    if args.smoke:
        uis.maxN.set(500)
        uis.runtime.set(0.01)
    if args.save_neuron_pos:
        uis.save_neuron_pos.set("True")
    if args.mode != "orig":
        uis.A1.set(0)

    params = uis.get_process_params()
    return sp3.net_setup(*params, plot_topo=args.plot_topo), params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLEOSim case study 3")

    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Run a short smoke test (set maxN to 500 and runtime to 0.01 s)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="orig",
        help="Select experiment mode: orig, OL, CL, or fit",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="numpy",
        help="brian2.prefs.codegen.target: numpy or cython",
    )
    parser.add_argument(
        "--Irr0_OL",
        type=float,
        default=0,
        help="Irradiance (mW/mm^2) for constant open-loop photostimulation",
    )
    parser.add_argument(
        "--maxIrr0",
        type=float,
        default=30,
        help="Maximum Irr0 value optic fiber can take",
    )
    parser.add_argument(
        "--R0",
        type=float,
        default=2,
        help="Optic fiber radius (in mm)",
    )

    # args from original interface
    parser.add_argument(
        "--maxN",
        type=int,
        default=10000,
        help="Choose the maximum number of neurons in the network (in the CA1 excitatory neurons group) :\nThe total number of neurons will be 3.32*N",
    )
    parser.add_argument(
        "--runtime", type=float, default=0.5, help="Duration of the simulation (s)"
    )
    parser.add_argument("--f1", type=float, default=2.5, help="Input frequency (Hz)")
    parser.add_argument(
        "--A1",
        type=float,
        default=1,
        help="Max current (nA) of model's square wave input",
    )
    parser.add_argument(
        "--save_neuron_pos",
        action="store_true",
        default=False,
        help="Save neuron positions as txt files",
    )

    # visualization
    parser.add_argument(
        "--plot_topo",
        action="store_true",
        default=False,
        help="Plot neuron and electrode positions",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        default=False,
        help="Show interactive plot windows after simulation",
    )

    # args for wrapping with CLEOSim

    # parser.add_argument("--model", required=True, help="Model type (resnet or alexnet)")
    # parser.add_argument("--niter", type=int, default=1000, help="Number of iterations")
    # parser.add_argument("--in_dir", required=True, help="Input directory with images")
    # parser.add_argument("--out_dir", required=True, help="Output directory with trained model")

    args = parser.parse_args()
    args.maxIrr0 = max(args.maxIrr0, args.Irr0_OL)
    with plt.style.context(["seaborn-paper"]):
        main(args)
