# %%
import ast
import os

import cleo.utilities
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Set working directory to the location of this file
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
data_dir = "aussel22-data"

# # %%
# def norm(signal):
#     return (signal - signal.mean()) / np.abs(signal).max()


# def band_powers(lfp, out_samp_rate=1000, fname=None):
#     """returns power over time (Tx6 array)"""
#     band_names = ["theta", "beta2", "gamma", "SWR"]
#     band_lim1 = [4, 23, 30, 150]
#     band_lim2 = [12, 30, 100, 250]
#     f, t, Sxx = signal.spectrogram(lfp, 1024, nperseg=None)
#     print(Sxx.shape)
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.pcolormesh(t, f, Sxx, cmap="magma")
#     # plt.ylim(0, 250)
#     for i_band in range(len(band_names))[:1]:
#         f_i_band = np.logical_and(f > band_lim1[i_band], f < band_lim2[i_band])
#         Sxx_for_band = Sxx[f_i_band, :].sum(axis=0)
#         # Sxx_for_band_rel = Sxx_for_band / Sxx[~f_i_band, :].sum(axis=0)
#         # Sxx_for_band_rel = Sxx_for_band / Sxx.sum(axis=0)
#         # Sxx_for_band_norm = norm(Sxx_for_band)
#         Sxx_for_band_norm = Sxx_for_band / Sxx_for_band.max()
#         # Sxx_for_band_rel_norm = norm(Sxx_for_band_rel)
#         # Sxx_for_band_rel_norm = Sxx_for_band_rel / Sxx_for_band_rel.max()
#         ax2.plot(t, Sxx_for_band_norm + i_band, label=band_names[i_band])
#     ax2.legend()


# for filename in os.listdir(data_dir):
#     file_path = os.path.join(data_dir, filename)
#     fig, ax = plt.subplots()
#     if not os.path.isfile(file_path):
#         continue
#     try:
#         data = np.loadtxt(file_path)
#     except ValueError:
#         # in case of printed list instead of numpy format
#         # don't use np.loadtxt
#         with open(file_path, "r") as f:
#             data = f.read()
#         data = np.array(ast.literal_eval(data))
#     ax.plot(data)
#     ax.set(title=filename)
#     # band_powers(data)

# plt.show()


# %%
def theta_power(
    lfp,
    fs=1024,
    theta_llim=4,
    theta_ulim=10,
    nperseg=2048,
    poverlap=7 / 8,
    window=("tukey", 0.25),
):
    if poverlap:
        noverlap = int(nperseg * poverlap)
    else:
        noverlap = None
    f, t, Sxx = signal.spectrogram(
        lfp, fs, nperseg=nperseg, noverlap=noverlap, window=window
    )
    f_i_band = np.logical_and(f >= theta_llim, f <= theta_ulim)
    return Sxx[f_i_band, :].sum(axis=0), t


thetas = []
labels = []
for filename in os.listdir(data_dir):
    if "LFP" not in filename:
        continue
    file_path = os.path.join(data_dir, filename)
    with open(file_path, "r") as f:
        data = f.read()
    data = np.array(ast.literal_eval(data))
    labels.append(filename)
    theta, t = theta_power(data)
    thetas.append(theta)
fig, ax = plt.subplots()
thetas = np.array(thetas).T
assert len(labels) == thetas.shape[1]
thetas /= thetas.max()
for i, label in enumerate(labels):
    ax.plot(t, thetas[:, i], label=label)
ax.legend()
# ax.axvline(35, c='k')
ax.set(xlim=(0, 35))

# %%
# make sure inputs look right after preprocessing
# no, looks like they are already preprocessed to be normalized firing rates
# so I need to disable that step for these files
from aussel_model.model.apply_input import get_FR

inputs_123 = []

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
for i, ax in enumerate(axs):
    filename = f"input_epi_wake_{i+1}.txt"
    file_path = os.path.join(data_dir, filename)
    inputs = np.loadtxt(file_path)
    inputs_123.append(inputs)

    ax.plot(data, label="raw", alpha=0.5)
    inputs_FR = get_FR(inputs, 1024)
    ax.plot(inputs_FR, label="FR", alpha=0.5)
    maxFR = 200
    inputs_norm = maxFR * inputs_FR / max(inputs_FR)
    ax.plot(inputs_norm, label="norm. FR", alpha=0.5)
    ax.set(title=filename)
    ax.legend()

# %%
# confirmed that the inputs are now loaded correctly without preprocessing
inputs = np.load("../val_healthy_results/input.npz")
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(inputs["inputs1"])
axs[1].plot(inputs["inputs2"])
axs[2].plot(inputs["inputs3"])
inputs.keys()
ax.plot(np.loadtxt("./aussel22-data/input_epi_wake_1.txt"))

# %%
import cleo

import aussel_model.model
import aussel_model.model.apply_input

theta_max = None

for model_type in ["epi", "healthy"]:
    lfp_orig = np.array(
        aussel_model.model.apply_input.lecture(f"../val_{model_type}_results/LFP.txt")[
            0
        ]
    )
    t_ms_orig = np.arange(len(lfp_orig)) / 1024 * 1000
    t_ms_rwslfp = np.load(f"../val_{model_type}_results/t_ms_rwslfp.npy")
    t_ms_tklfp = np.load(f"../val_{model_type}_results/t_ms_tklfp.npy")
    tklfp = np.load(f"../val_{model_type}_results/tklfp.npy")
    rwslfp = np.load(f"../val_{model_type}_results/rwslfp.npy")

    avg_input = np.mean(inputs_123, axis=0)
    t_ms_input = np.arange(len(avg_input)) / 1024 * 1000
    avg_input = avg_input[t_ms_input < 35000]
    t_ms_input = t_ms_input[t_ms_input < 35000]
    # avg_input_norm = avg_input / np.abs(avg_input[t_ms_input < 5000]).max()

    plt.style.use("default")
    plt.rcParams.update({"svg.fonttype": "none", "savefig.dpi": 300})

    for t_ms, lfp, name in [
        (t_ms_orig, lfp_orig, "sclfp"),
        (t_ms_rwslfp, rwslfp, "rwslfp"),
        (t_ms_tklfp, tklfp, "tklfp"),
        (t_ms_input, avg_input, "avg SEEG input"),
    ]:
        aspect = 1.5
        height = 4
        fig, (ax1, ax2) = plt.subplots(
            1, 2, layout="tight", figsize=(aspect * height, height)
        )
        lfp_norm = lfp / np.abs(lfp[t_ms < 5000]).max()
        ax1.plot(t_ms / 1000, lfp_norm, c="#c500cc", lw=1, rasterized=True)
        ax1.set(
            ylabel=f"Normalized {name.upper()}", xlabel="Time (s)", ylim=(-3.5, 8.5)
        )
        # Aussel used 4000 npserseg and 3900 noverlap on fs=1024 data. and default window
        win_width = 4000 * 1000 / 1024
        nperseg = np.searchsorted(t_ms, win_width)
        # print(f"win_width = {win_width}, nperseg = {nperseg}")
        poverlap = 3900 / 4000
        theta, t = theta_power(lfp_norm, nperseg=nperseg, poverlap=poverlap)
        if not theta_max:
            theta_max = theta.max()
        ax2.plot(t, theta / theta_max, c="#c500cc", rasterized=True)
        ax2.set(
            ylabel="Normalized theta band power",
            xlabel="Time (s)",
            xlim=(0, 35),
            ylim=(-0.05, 1.05),
        )

        fig.suptitle(model_type)
        fig.savefig(
            f"../results/val-{model_type}-{name}.svg",
            bbox_inches="tight",
            transparent=True,
        )

# %%
t_ms_orig

# %%
