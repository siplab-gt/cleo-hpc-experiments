#!python
import os
from pathlib import Path

# import sys
import argparse
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

import ldsctrlest as lds
import ldsctrlest.gaussian as glds


def main(args):
    u = []
    z = []
    for results_dir in args.results_dirs:
        results_dir = Path(results_dir)
        in_npz = np.load(results_dir / "input.npz")
        u_opto = in_npz["Irr0_mW_per_mm2"]
        # inexplicably, we have some missed samples
        # maybe dt was occasionally just a little off, so t % dt != 0?
        # or a hiccup in calling the network operation?
        # we can either skip these 2 ms samples, divide the data into 
        # segments, or interpolate
        # I'll divide into segments
        u_opto = u_opto.reshape((1, -1))
        t_ms = in_npz["t_opto_ms"]
        assert np.all(np.diff(t_ms) < 3)
        t_ms_tklfp = np.load(results_dir / "t_ms_tklfp.npy")
        assert np.all(t_ms == t_ms_tklfp)
        z_tklfp = np.load(results_dir / "tklfp.npy")
        z_tklfp = z_tklfp.reshape((1, -1))
        assert u_opto.shape[1] == z_tklfp.shape[1], (u_opto.shape[1], z_tklfp.shape[1])
        # need transpose so time is in columns
        i_skip_prev = 0
        for i_skip in np.where([np.isclose(np.diff(t_ms), 2)])[1]:
            u.append(u_opto[:, i_skip_prev:i_skip])
            z.append(z_tklfp[:, i_skip_prev:i_skip])
            i_skip_prev = i_skip + 1
        u.append(u_opto[:, i_skip_prev:])
        z.append(z_tklfp[:, i_skip_prev:])
        print(f'Found {len(u)} segments in {results_dir}')
        u = [u_opto]
        z = [z_tklfp]
    if args.dry_run:
        return

    n_x_fit = args.nx  # latent dimensionality of system
    n_h = 50  # size of block Hankel data matrix
    dt = 0.001  # timestep (in seconds)
    u_uml = lds.UniformMatrixList(u, free_dim=2)
    z_uml = lds.UniformMatrixList(z, free_dim=2)
    ssid = glds.FitSSID(n_x_fit, n_h, dt, u_uml, z_uml)
    fit, sing_vals = ssid.Run(lds.SSIDWt.kMOESP)

    # EM
    if args.iterEM > 0:
        calc_dynamics = True  # calculate dynamics (A, B mats)
        calc_Q = True  # calculate process noise cov (Q)
        calc_init = True  # calculate initial conditions
        calc_output = True  # calculate output (C)
        calc_measurement = True  # calculate output noise (R)
        max_iter = args.iterEM
        tol = args.tolEM

        em = glds.FitEM(fit, u_uml, z_uml)

        start = time.perf_counter()
        fit = em.Run(
            calc_dynamics,
            calc_Q,
            calc_init,
            calc_output,
            calc_measurement,
            max_iter,
            tol,
        )
        stop = time.perf_counter()
        print(f"Finished EM fit in {(stop-start)*1000} ms.")

    n_samp_imp = int(np.ceil(0.1 / dt))
    t_imp = np.arange(0, n_samp_imp * dt, dt)

    # compare fit to original without state noise
    sys_hat = glds.System(fit)
    sys_hat.Q = np.zeros_like(sys_hat.Q)
    y_hat, x_hat, _ = sys_hat.simulate_block(u)
    y_imp_hat = sys_hat.simulate_imp(n_samp_imp)

    # SSID plot singular values & impulse response

    fig, axs = plt.subplots(1, 2)
    axs[0].semilogy(sing_vals[:n_h], "-o", color=[0.5, 0.5, 0.5])
    axs[0].semilogy(sing_vals[:n_h], color="k", linewidth=1)
    axs[0].set(ylabel="Singular Values", xlabel="Singular Value Index")

    l2 = axs[1].plot(t_imp, y_imp_hat[0].T, "-", c="#C500CC", linewidth=2)
    axs[1].set(ylabel="Impulse Response (a.u.)", xlabel="Time (s)")
    fig.tight_layout()
    fig

    # %%
    # SSID plot var explained
    for z_trial, y_hat_trial, u_trial in zip(z, y_hat, u):
        var = np.var(z_trial, axis=(0, 1))
        var_not_explnd = np.var(z_trial - y_hat_trial, axis=(0, 1))
        pve = 1 - var_not_explnd / var

        fig, axs = plt.subplots(2, 1, figsize=(6, 3), sharex=True)

        t = np.arange(z_trial.shape[1]) / 1000
        axs[0].plot(t, z_trial[0, :], "k-")
        axs[0].plot(t, y_hat_trial[0, :], "-", c="#C500CC", linewidth=2)
        axs[0].legend(["measurement", "fit"])
        axs[0].set(
            ylabel=f"TKLFP (μV)",
            title=f"proportion var explained (training): {pve:0.3f}",
        )

        axs[1].plot(t, u_trial.T, "k")
        axs[1].set(ylabel="Input (a.u.)", xlabel="Time (s)")

        fig.tight_layout()
    plt.show()

    # with open(args.out,'wb') as fh:
    #     pickle.dump(fit, fh)
    attrs_to_save = {
        var: getattr(fit, var)
        for var in [
            "n_u",
            "n_x",
            "n_y",
            "dt",
            "A",
            "B",
            "C",
            "d",
            "g",
            "m",
            "Q",
            "x0",
            "P0",
            "R",
        ]
    }
    np.savez(args.out, **attrs_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit linear model to data")

    parser.add_argument("results_dirs", nargs="+", metavar="results_dir")
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--iterEM", type=int, default=0, help="max_iter for EM")
    parser.add_argument("--tolEM", type=float, default=1e-2, help="tol for EM")
    parser.add_argument("--nx", type=int, default=4, help="num hidden states for fit")
    parser.add_argument(
        "--out", type=str, default="results/fit.npz", help="where to store fit"
    )

    main(parser.parse_args())
