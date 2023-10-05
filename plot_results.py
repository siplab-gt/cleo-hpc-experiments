#!python
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt


def plot_lfp(path):
    fig, ax = plt.subplots()

    ref_fname = os.path.join(path, 'ref.npy')
    if os.path.exists(ref_fname):
        ax.plot(np.load(ref_fname), lw=2, c='gray')

    lfp = np.load(os.path.join(path, "tklfp.npy"))
    ax.plot(lfp, c='black', )
    plt.title("TKLFP")
    plt.ylabel("μV")
    plt.xlabel("ms")

    if os.path.exists(ref_fname):
        plt.legend(['reference', 'measured'])


def plot_input(path):
    npz = np.load(os.path.join(path, "input.npz"))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(npz["t_s"]*1000, npz["inputs1"])
    ax1.set(title="external current", ylabel="$I_{ext}$ (nA)")
    try:
        ax2.step(npz["t_opto_ms"], npz["Irr0_mW_per_mm2"], where='post')
        ax2.set(title="optogenetic input", xlabel="t (ms)", ylabel="$Irr_0$ (mW/mm$^2$)")
    except KeyError:
        pass

if __name__ == "__main__":
    plot_lfp(sys.argv[1])
    plot_input(sys.argv[1])
    plt.show()
