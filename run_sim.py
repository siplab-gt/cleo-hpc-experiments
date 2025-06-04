#! python
from __future__ import annotations

import argparse
import os
import shutil
import time

import cleo
import cleo.utilities
import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wslfp
from brian2 import Network, StateMonitor, mm, mm2, ms, mwatt, prefs, uvolt
from cleo.ephys import Probe, RWSLFPSignalFromPSCs, TKLFPSignal
from pyvirtualdisplay import Display
from scipy.linalg import solve_discrete_are

try:
    import ldsctrlest.gaussian as glds
except ModuleNotFoundError:
    print("Warning: ldsctrlest not installed, so LQR control will not work")

from plot_results import plot_input, plot_lfp

display = Display()
display.start()
print(f"display.is_alive() = {display.is_alive()}")
from aussel_model.interface import user_interface_simple as uis
from aussel_model.model import single_process3 as sp3


def main(args):
    setup_start = time.time()
    prefs.codegen.target = args.target
    (net, all_ngs, elec_pos), params = setup_aussel_net(args)
    path = params[23]
    all_ngs_exc = [area[0][0] for area in all_ngs]
    all_ngs_inh = [area[1][0] for area in all_ngs]

    assign_coords(all_ngs)
    sim = cleo.CLSimulator(net)

    n_opto_col = 10
    n_opto_tot = 2 * n_opto_col
    config_processor(args, sim, n_opto_tot, path)

    lfp_signals = [
        TKLFPSignal(name="lfp"),
        RWSLFPSignalFromPSCs(amp_func=wslfp.aussel18, name="rwslfp"),
    ]
    if args.sclfp:
        lfp_signals.append(
            RWSLFPSignalFromPSCs(
                amp_func=wslfp.aussel18,
                name="sclfp",
                wslfp_kwargs={"alpha": 1, "tau_ampa_ms": 0, "tau_gaba_ms": 0},
            )
        )
    # saclfp = RWSLFPSignalFromPSCs(
    #     amp_func=wslfp.aussel18, name="saclfp", wslfp_kwargs={}
    # )
    # use same electrode coordinates, with the same 150um scale
    probe = Probe(elec_pos * 0.15 * mm, lfp_signals, save_history=True)
    for ng_exc, ng_inh in zip(all_ngs_exc, all_ngs_inh):
        orntn = orntn_for_ng(ng_exc)
        mean_orntn = np.mean(orntn, axis=0, keepdims=True)
        # inh groups don't have orientation information stored, so we will approximate
        # with the average orientation for the exc neurons in each region
        sim.inject(
            probe,
            ng_exc,
            tklfp_type="exc",
            orientation=orntn,
            Iampa_var_names=["I_SynE", "I_SynExt"],
            Igaba_var_names=["I_SynI"],
        )
        sim.inject(probe, ng_inh, tklfp_type="inh", orientation=mean_orntn)
    fibers = None
    if args.mode not in ["orig", "val-epi", "val-healthy"]:
        light_model = cleo.light.fiber473nm()
        light_model.R0 = args.R0 * mm  # bigger fiber radius
        light_model.K *= args.Kfactor  # alter absorbance
        light_model.S *= args.Sfactor  # alter scattering
        coords = np.zeros((n_opto_tot, 3)) * mm
        drctn = np.zeros((n_opto_tot, 3))
        # set coords, direction
        coords[:n_opto_col, :2] = [2.5, -6] * mm
        coords[:n_opto_col, 2] = np.linspace(0.5, 14.5, n_opto_col, endpoint=True) * mm
        drctn[:n_opto_col] = (0, 1, 0)
        coords[n_opto_col:, :2] = [2.5, -7] * mm
        coords[n_opto_col:, 2] = np.linspace(0.5, 14.5, n_opto_col, endpoint=True) * mm
        drctn[n_opto_col:] = (-1, 0, 0)
        opsin = cleo.opto.chr2_4s()
        sim.inject(opsin, all_ngs_exc[0], all_ngs_inh[0])
        fibers = cleo.light.Light(
            name="fibers",
            coords=coords,
            direction=drctn,
            light_model=light_model,
            save_history=True,
            max_value=args.maxIrr0 * mwatt / mm2,
        )
        sim.inject(fibers, all_ngs_exc[0], all_ngs_inh[0])
    # mon_Iopto = StateMonitor(all_ngs_exc[0], 'Iopto', record=True)
    # sim.network.add(mon_Iopto)
    plot_viz(args, all_ngs_exc, all_ngs_inh, probe, fibers)

    print(f"Setup time: {(time.time() - setup_start)} seconds")

    if args.runtime > 0:
        sp3.run_process(net, all_ngs, elec_pos, *params)

        # fig, ax = plt.subplots()
        # ax.plot(mon_Iopto.t, mon_Iopto.Iopto.T)

        save_lfp(path, *lfp_signals)
        plot_lfp(path)
        save_input(path, fibers)
        plot_input(path)

    uis.aborted = False
    if not args.no_save:
        uis.save_plots()
    if args.show_plots:
        plt.show()
    if args.no_save:
        shutil.rmtree(path)


def config_processor(args, sim, n_opto, path):
    dt_samp = 1 * ms
    t_start_ms = 100
    t_stop_ms = 300
    t_trial_ms = 400

    if args.ref:
        ref = np.tile(np.load(args.ref), args.n_trials)
        np.save(os.path.join(path, "ref.npy"), ref)

    if args.mode in ["orig", "val-epi", "val-healthy"]:
        # this is equivalent to the RecordOnlyProcessor
        my_process = lambda state, t_samp: ({}, t_samp)

    elif args.mode == "OLconst":

        def my_process(state, t_samp):
            if t_start_ms <= t_samp / ms < t_stop_ms:
                opto_val = args.Irr0_OL * mwatt / mm2
            else:
                opto_val = 0 * mwatt / mm2
            return {"fibers": opto_val}, t_samp / ms

    elif args.mode == "OLnaive":
        u = -ref / np.max(np.abs(ref)) * args.maxIrr0
        u[u < 0] = 0

        def my_process(state, t_samp):
            return {"fibers": u[int(t_samp / ms)] * mwatt / mm2}, t_samp

    elif args.mode == "OLLQR":
        # compute stimulus beforehand using model fit
        gsys = load_fit_sys(path, args)
        sys2sim = gsys.copy()
        ctrlr = glds.Controller(gsys, u_lb=0, u_ub=args.maxIrr0)
        ctrlr.Kc = lqr_gain(gsys, args.r)
        sim.u = np.empty_like(ref)
        for t, yref in enumerate(ref):
            ctrlr.y_ref = yref
            sim.u[t] = ctrlr.ControlOutputReference(sys2sim.y)[0, 0]
            sys2sim.Simulate(sim.u[t])

        def my_process(state, t_samp):
            opto_val = sim.u[int(t_samp / ms)] * mwatt / mm2
            return {"fibers": opto_val}, t_samp

    elif args.mode == "fit":
        n_tot = int(args.runtime * 1000)
        on_off = np.array([])
        # alternate between on and off periods
        while len(on_off) < args.runtime * 1000:
            n_on, n_off = np.ceil(200 + 50 * np.random.randn(2))
            if n_on < 0:
                n_on = 0
            if n_off < 0:
                n_off = 0
            on_off = np.concatenate([on_off, np.ones(int(n_on)), np.zeros(int(n_off))])
        # when on:
        # one side of a normal distribution. Max is 3 st devs away
        u_rand = np.abs(args.maxIrr0 / 3 * np.random.randn(n_tot))
        sim.u = on_off[:n_tot] * u_rand

        def my_process(state, t_samp):
            opto_val = sim.u[int(t_samp / ms)] * mwatt / mm2
            return {"fibers": opto_val}, t_samp

    elif args.mode == "LQR":
        gsys = load_fit_sys(path, args)
        ctrlr = glds.Controller(gsys, u_lb=0, u_ub=args.maxIrr0)
        ctrlr.Kc = lqr_gain(gsys, args.r)
        sim.ref = ref
        shutil.copy(args.ref, os.path.join(path, "ref.npy"))
        sim.ctrlr = ctrlr

        def my_process(state, t_samp):
            lfp_uV = state["Probe"]["lfp"]
            lfp1 = lfp_uV[:144].mean()
            lfp2 = lfp_uV[144:288].mean()
            # assuming regular samples, can us t_ms directly as index
            sim.ctrlr.y_ref = sim.ref[int(t_samp / ms)]
            opto_val = sim.ctrlr.ControlOutputReference(lfp2 - lfp1)[0, 0] * mwatt / mm2
            return {"fibers": opto_val}, t_samp + 3 * ms

    elif args.mode == "MPC":
        # load model
        fit = dict(np.load(args.fit))
        # initial state and estimate uncertainty
        sim.P = fit["P0"]
        # don't know why this needs to be a float
        sim.Q = fit["Q"]
        sim.R = fit["R"].flatten()[0]
        sim.A = fit["A"]
        sim.B = fit["B"]
        sim.C = fit["C"]
        n_x = sim.A.shape[0]
        sim.x_est = np.zeros(n_x)  # initial state estimate
        sim.ref = ref
        sim.optimal_u = 0.0

        # implement using cvxpy
        horizon = 50
        # pad reference so horizon doesn't go out of bounds
        sim.ref = np.pad(sim.ref, (0, horizon), mode="edge")
        sim.uopt = cp.Variable((1, horizon), nonneg=True)
        sim.xopt = cp.Variable((n_x, horizon + 1))  # x traj including initial state
        # initialize with zeros
        sim.uopt.value = np.zeros((1, horizon))
        sim.xopt.value = np.zeros((n_x, horizon + 1))

        sim.yopt = sim.C @ sim.xopt[:, 1:]  # predicted output for horizon
        sim.x0opt = cp.Parameter(n_x)  # initial state
        sim.yrefopt = cp.Parameter((1, horizon))
        cost = cp.sum_squares(sim.yopt - sim.yrefopt) + (
            args.r * cp.sum_squares(sim.uopt)
        )
        constraints = [
            sim.xopt[:, 0] == sim.x0opt,  # initial state
            # system dynamics
            sim.xopt[:, 1:] == sim.A @ sim.xopt[:, :-1] + sim.B @ sim.uopt,
            sim.uopt <= args.maxIrr0,  # input constraints
        ]
        sim.prob = cp.Problem(cp.Minimize(cost), constraints)
        print("MPC problem is DCP?", sim.prob.is_dcp(dpp=False))
        print("MPC problem is DPP?", sim.prob.is_dcp(dpp=True))
        sim.solve_times = []

        def my_process(state, t_samp):
            # get measurement
            lfp_uV = state["Probe"]["lfp"] / uvolt
            lfp1 = lfp_uV[:144].mean()
            lfp2 = lfp_uV[144:288].mean()
            y = lfp2 - lfp1

            # kalman filter to get xhat_t|t
            P_pred = sim.A @ sim.P @ sim.A.T + sim.Q
            K = P_pred @ sim.C.T @ np.linalg.inv(sim.C @ P_pred @ sim.C.T + sim.R)
            x_pred = sim.xopt[:, 1].value
            y_err = y - sim.C @ x_pred
            # updated state and covariance estimates
            sim.x0opt.value = x_pred + K @ y_err
            sim.P = (np.eye(n_x) - K @ sim.C) @ P_pred

            # assuming regular samples, can us t_ms directly as index
            t_index = int(t_samp / ms)
            # ref from t+1 to t+horizon (since y at current step can't be helped)
            sim.yrefopt.value = sim.ref[t_index + 1 : t_index + 1 + horizon].reshape(
                (1, horizon)
            )

            sim.prob.solve(warm_start=True)
            sim.solve_times.append(sim.prob.solver_stats.solve_time)

            # want to use uopt[:, 0] but Cleo v0.18.1 doesn't broadcast unless scalar
            return {"fibers": sim.uopt[0, 0].value * mwatt / mm2}, t_samp + 6 * ms
    elif args.mode == "OLMPC":
        raise NotImplementedError(
            "This was done previously with the Julia setup, not yet with cvxpy. "
            "The idea is running MPC over the whole trial with no feedback---"
            "the steel-manned version of OL control."
        )

    # need to subclass so it's concrete
    class MyLIOP(cleo.ioproc.LatencyIOProcessor):
        def process(self, state, t_samp):
            return my_process(state, t_samp)

    proc = MyLIOP(dt_samp)
    sim.set_io_processor(proc)


def load_fit_sys(path, args) -> glds.System:
    fit = dict(np.load(args.fit))
    # save system fit
    shutil.copy(args.fit, os.path.join(path, "fit.npz"))
    sys = glds.System(fit.pop("n_u"), fit.pop("n_x"), fit.pop("n_y"), fit.pop("dt"))
    for k, v in fit.items():
        setattr(sys, k, v)
    return sys


def lqr_gain(sys: glds.System, r: float):
    Q = sys.C.T @ sys.C
    A, B = sys.A, sys.B
    P = solve_discrete_are(A, B, Q, r)
    return np.linalg.inv(r + B.T @ P @ B) @ (B.T @ P @ A)


def plot_viz(args, all_ngs_exc, all_ngs_inh, probe, fibers=None):
    if args.opto_slice:
        old_coords, old_dir = fibers.coords, fibers.direction
        fibers.coords = fibers.coords[[4, 14]]
        fibers.direction = fibers.direction[[4, 14]]
    devices = [
        (probe, {"size": 5, "color": (0.1, 0.1, 0.1, 0.5), "marker": "."}),
    ]
    if fibers:
        devices.append((fibers, {"n_points": 3e4, "intensity": 0.6}))
    colors_exc = ["#fb9a99", "#fdbf6f", "#b2df8a", "#cab2d6"]
    colors_inh = ["#e31a1c", "#ff7f00", "#33a02c", "#6a3d9a"]
    colors = colors_exc + colors_inh
    fig, ax = cleo.viz.plot(
        *all_ngs_exc,
        *all_ngs_inh,
        zlim=(6, 8.5),
        colors=colors,
        invert_z=False,
        devices=devices,
        scatterargs={
            "rasterized": True,
            "alpha": 0.8,
            "marker": ".",
            "s": 2 * 10000 / args.maxN,
        },
        figsize=(4, 4),
        axis_scale_unit=mm,
    )
    ax.set(zticks=[7, 8, 9])
    ax.view_init(60, -125)
    ax.get_legend().remove()
    if args.opto_slice:
        fibers.coords, fibers.direction = old_coords, old_dir


def assign_coords(all_ngs):
    # all_ngs is 4 x 2 x 1 nested list.
    # 4 areas, 2 types, and I don't know why the last level:
    # aussel_model/model_files/preparation.py:337
    for i_area in range(4):  # EC, DG, CA3, CA1
        for i_type in range(2):  # exc, inh
            ng = all_ngs[i_area][i_type][0]
            cleo.coords.assign_xyz(ng, ng.x_soma / mm, ng.y_soma / mm, ng.z_soma / mm)


def orntn_for_ng(ng):
    xyz_dendrite = np.column_stack(
        (ng.x_dendrite / mm, ng.y_dendrite / mm, ng.z_dendrite / mm)
    )
    xyz_soma = np.column_stack((ng.x_soma / mm, ng.y_soma / mm, ng.z_soma / mm))
    assert xyz_dendrite.shape == xyz_soma.shape == (ng.N, 3)
    return xyz_dendrite - xyz_soma


def save_lfp(
    path,
    lfp: TKLFPSignal,
    rwslfp: RWSLFPSignalFromPSCs,
    sclfp: RWSLFPSignalFromPSCs = None,
):
    # imitate method from Aussel 2018: take average signal from one cylinder
    # of contacts and subtract from the other
    lfp_vals_and_names = [(lfp.lfp / uvolt, "tklfp"), (rwslfp.lfp, "rwslfp")]
    if sclfp:
        lfp_vals_and_names.append((sclfp.lfp, "sclfp"))
    for lfp_vals, signal_type in lfp_vals_and_names:
        lfp1 = lfp_vals[:, :144].mean(axis=1)
        lfp2 = lfp_vals[:, 144:288].mean(axis=1)
        fname = os.path.join(path, f"{signal_type}.npy")
        np.save(fname, lfp2 - lfp1)
        t_fname = os.path.join(path, f"t_ms_{signal_type}.npy")
        np.save(t_fname, lfp.t / ms)


def save_input(path, fibers: cleo.light.Light):
    if fibers is None:
        return
    fname = os.path.join(path, "input.npz")
    npzfile = np.load(fname)
    Irr0 = fibers.irradiance_[:, 0]  # just get one channel, since they're identical
    np.savez_compressed(fname, Irr0_mW_per_mm2=Irr0, t_opto_ms=fibers.t / ms, **npzfile)


# %%
# l is *1024/1000 to convert from their 1024 Hz samples to ms
def gp_noise(in1, mu=0.2, σ=0.1, l=30 * 1024 / 1000):
    """assuming no units, will be nA"""
    t = np.arange(len(in1))
    t1, t2 = np.meshgrid(t, t)
    Σ = σ**2 * np.exp(-((t2 - t1) ** 2) / (2 * l**2))
    rng = np.random.default_rng()
    noised = rng.multivariate_normal(in1 + mu, Σ)
    return noised * 5 / 6


# a = np.zeros(400)
# a[100:300] = 1
# plt.plot(gp_noise(a))
# plt.plot(gp_noise(np.zeros_like(a)))
# %%


def setup_aussel_net(args) -> tuple[Network, list]:
    uis.f1.set(args.f1)
    kwargs = {}

    if args.mode == "val-epi":
        # pathological parameters from Aussel 2022, Fig 5
        uis.sclerosis.set(0.6)
        uis.sprouting.set(0.8)
        uis.Ek.set(-90)
        uis.tau_Cl.set(0.5)
    if args.mode in ["val-epi", "val-healthy"]:
        uis.pmono.set(0.3)
        uis.functional_co.set("wake")
        uis.CAN.set("wake")
        uis.input_type.set("custom")
        input_basename = "validation/aussel22-data/input_epi_wake_?.txt"
        uis.in_file_1.set(input_basename.replace("?", "1"))
        uis.in_file_2.set(input_basename.replace("?", "2"))
        uis.in_file_3.set(input_basename.replace("?", "3"))
        kwargs["preprocess_inputs"] = False

    uis.maxN.set(args.maxN)
    uis.runtime.set(args.runtime)
    if args.save_neuron_pos:
        uis.save_neuron_pos.set("True")
    if args.mode != "orig":
        uis.A1.set(0)
    if args.noise:
        kwargs["noise_adder"] = gp_noise
    kwargs["plot_topo"] = args.plot_topo

    params = uis.get_process_params()
    return (
        sp3.net_setup(*params, **kwargs),
        params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cleo case study 3")

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
        help="Select experiment mode: orig, OLconst, OLLQR, LQR, MPC, fit, val-epi, or val-healthy",
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
        default=75,
        help="Maximum Irr0 value optic fiber can take. 75 default is from Cardin et al., 2010",
    )
    parser.add_argument(
        "--fit",
        type=str,
        default=None,
        help="For CL control: .npz file containing system parameters previously fit",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="For OL/CL control: .npy file containing TKLFP waveform to evoke",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="For OL/CL control: number of trials to perform",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=1e-3,
        help="For CL control: input penalty in quadratic cost function",
    )
    parser.add_argument(
        "--R0", type=float, default=0.2, help="Optic fiber radius (in mm)"
    )
    parser.add_argument(
        "--Kfactor",
        type=float,
        default=0.1,
        help="Factor by which to multiply default absorbance coefficient (K)",
    )
    parser.add_argument(
        "--Sfactor",
        type=float,
        default=0.1,
        help="Factor by which to multiply default scattering coefficient (S)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        default=False,
        help="Results folder will be deleted at the end",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        default=False,
        help="Add noise to the Iext input",
    )

    # args from original interface
    parser.add_argument(
        "--maxN",
        type=int,
        default=10000,
        help="Choose the maximum number of neurons in the network (in the CA1 excitatory neurons group) :\nThe total number of neurons will be 3.32*N",
    )
    parser.add_argument(
        "--runtime", type=float, default=0, help="Duration of the simulation (s)"
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
    parser.add_argument(
        "--opto_slice",
        action="store_true",
        default=False,
        help="Whether to only plot fibers in the slice visualized",
    )
    parser.add_argument(
        "--sclfp",
        action="store_true",
        default=False,
        help="Run with simply summed currents (no weighting or delay) through Cleo; should be identical to original",
    )

    # args for wrapping with cleo

    # parser.add_argument("--model", required=True, help="Model type (resnet or alexnet)")
    # parser.add_argument("--niter", type=int, default=1000, help="Number of iterations")
    # parser.add_argument("--in_dir", required=True, help="Input directory with images")
    # parser.add_argument("--out_dir", required=True, help="Output directory with trained model")

    args = parser.parse_args()
    if not args.show_plots:
        matplotlib.use("Agg")
    args.maxIrr0 = max(args.maxIrr0, args.Irr0_OL)
    if args.ref:
        ref = np.load(args.ref)
        assert len(ref) == 400
        args.runtime = len(ref) * args.n_trials / 1000
    if args.smoke:
        args.maxN = 500
        args.runtime = 0.01
        args.no_save = True

    cleo.utilities.style_plots_for_paper()
    try:
        main(args)
    finally:
        print("shutting down virtual display")
        display.stop()

# %%
