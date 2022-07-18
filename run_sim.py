import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from brian2 import Network, mm, ms

import cleosim
from cleosim.electrodes import Probe, TKLFPSignal
from cleosim import opto

from aussel_model.model import single_process3 as sp3
from aussel_model.interface import user_interface_simple as uis


def main(args):
    setup_start = time.time()
    (net, all_ngs, elec_pos), params = setup_aussel_net(args)
    all_ngs_exc = [area[0][0] for area in all_ngs]
    all_ngs_inh = [area[1][0] for area in all_ngs]

    assign_coords(all_ngs)
    sim = cleosim.CLSimulator(net)

    sim.set_io_processor(cleosim.processing.RecordOnlyProcessor(1))

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
    op_int = opto.OptogeneticIntervention(
        'opto',
        opto.FourStateModel(opto.ChR2_four_state),
        opto.default_blue,
        (3, -3, 7.5)*mm,
        (-1, 0, 0)
    )

    if args.viz:
        cleosim.visualization.plot(
            *all_ngs_exc, *all_ngs_inh, invert_z=False, devices_to_plot=[probe, op_int]
        )

    print(f"Setup time: {(time.time()-setup_start)} seconds")

    sp3.run_process(net, all_ngs, elec_pos, *params)

    plot_lfp(lfp)
    plot_input(path=params[23], op_int=op_int)

    uis.aborted = False
    uis.save_plots()
    if args.show_plots:
        plt.show()


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


def plot_lfp(lfp: TKLFPSignal):
    # imitate method from Aussel 2018: take average signal from one cylinder
    # of contacts and subtract from the other
    lfp1 = lfp.lfp_uV[:, :144].mean(axis=1)
    lfp2 = lfp.lfp_uV[:, 144:288].mean(axis=1)
    plt.figure()
    plt.plot(lfp2 - lfp1)
    plt.title("TKLFP")
    plt.ylabel("$\mu$V")
    plt.xlabel("ms")


def plot_input(path, op_int):
    npzfile = np.load(os.path.join(path, 'input.npz'))
    t_s, inputs1 = [npzfile[v] for v in ['t_s', 'inputs1']]
    plt.figure()
    plt.title('Input')
    plt.plot(t_s, inputs1, label='endogenous input')
    plt.figure()


def setup_aussel_net(args) -> Network:
    uis.maxN.set(args.maxN)
    uis.runtime.set(args.runtime)
    uis.f1.set(args.f1)
    if args.smoke:
        uis.maxN.set(500)
        uis.runtime.set(0.01)
    if args.save_neuron_pos:
        uis.save_neuron_pos.set("True")

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
    parser.add_argument(
        "--f1", type=float, default=2.5, help="Input frequency (Hz)"
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
        "--viz",
        action="store_true",
        default=False,
        help="Visualize using CLEOSim utilities",
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
    main(args)
