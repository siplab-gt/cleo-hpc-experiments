#!python
import os
from pathlib import Path
# import sys
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import ldsctrlest as lds
import ldsctrlest.gaussian as glds


def main(args):
    u = []
    z = []
    for results_dir in args.results_dirs:
        results_dir = Path(results_dir)
        in_npz = np.load(results_dir / 'input.npz')
        u_opto = in_npz['Irr0_mW_per_mm2'][1:]
        u_opto = u_opto.reshape((1, -1))
        # need transpose so time is in columns
        u.append(u_opto)
        z_tklfp = np.load(results_dir / 'tklfp.npy')
        z_tklfp = z_tklfp.reshape((1, -1))
        z.append(z_tklfp)
        assert u_opto.shape[1] == z_tklfp.shape[1]
    if args.dry_run:
        return

    n_x_fit = 4  # latent dimensionality of system
    n_h = 50  # size of block Hankel data matrix
    dt = 0.001  # timestep (in seconds)
    u_uml = lds.UniformMatrixList(u, free_dim=2)
    z_uml = lds.UniformMatrixList(z, free_dim=2)
    ssid = glds.FitSSID(n_x_fit, n_h, dt, u_uml, z_uml)
    fit, sing_vals = ssid.Run(lds.SSIDWt.kMOESP)

    # EM
    if False:
        calc_dynamics = True  # calculate dynamics (A, B mats)
        calc_Q = True  # calculate process noise cov (Q)
        calc_init = True  # calculate initial conditions
        calc_output = True  # calculate output (C)
        calc_measurement = True  # calculate output noise (R)
        max_iter = args.iterEM
        tol = 1e-2

        em = glds.FitEM(fit, u_uml, z_uml)

        start = time.perf_counter()
        fit = em.Run(
            calc_dynamics, calc_Q, calc_init, calc_output, calc_measurement, max_iter, tol
        )
        stop = time.perf_counter()
        print(f"Finished EM fit in {(stop-start)*1000} ms.")

    n_samp_imp = int(np.ceil(0.1 / dt))
    t_imp = np.arange(0, n_samp_imp * dt, dt)

    # compare fit to original without state noise
    sys_hat = glds.System(fit)
    sys_hat.Q = np.zeros_like(sys_hat.Q)
    y_hat, x_hat, _ = sys_hat.simulate_block(u)
    impulse = False
    if impulse:
        y_imp_hat = sys_hat.simulate_imp(n_samp_imp)

    # SSID plot singular values & impulse response

    fig, axs = plt.subplots(1, 2)
    axs[0].semilogy(sing_vals[:n_h], "-o", color=[0.5, 0.5, 0.5])
    axs[0].semilogy(sing_vals[:n_h], color="k", linewidth=2)
    axs[0].set(ylabel="Singular Values", xlabel="Singular Value Index")

    if impulse:
        l2 = axs[1].plot(t_imp, y_imp_hat[0].T, "-", c=[0.5, 0.5, 0.5], linewidth=2)
        axs[1].set(ylabel="Impulse Response (a.u.)", xlabel="Time (s)")
        fig.tight_layout()
    fig

    # %%
    # SSID plot var explained
    for z_trial, y_hat_trial, u_trial in zip(z, y_hat, u):
        var = np.var(z_trial, axis=(0, 1))
        var_not_explnd = np.var(z_trial - y_hat_trial, axis=(0, 1))
        pve = 1 - var_not_explnd / var

        fig, axs = plt.subplots(2, 1, figsize=(6, 3))

        t = np.arange(z_trial.shape[1])/1000
        axs[0].plot(t, z_trial[0, :], "k-")
        axs[0].plot(t, y_hat_trial[0, :], "-", c="gray", linewidth=2)
        axs[0].legend(["measurement", "fit"])
        axs[0].set(
            ylabel=f"TKLFP (μV)",
            xlabel="Time (s)",
            title=f"proportion var explained (training): {pve:0.3f}",
        )

        axs[1].plot(t, u_trial.T, "k")
        axs[1].set(ylabel="Input (a.u.)", xlabel="Time (s)")

        fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit linear model to data")

    parser.add_argument('results_dirs', nargs='+', metavar='results_dir')
    parser.add_argument('--dry_run', action='store_true', default=False)
    parser.add_argument('--iterEM', type=int, default=100, help="max_iter for EM")

    main(parser.parse_args())