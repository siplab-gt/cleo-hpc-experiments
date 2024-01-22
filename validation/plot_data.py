# %%
import os
import numpy as np
import ast
import matplotlib.pyplot as plt
from scipy import signal

# Set working directory to the location of this file
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)

# %%

data_dir = "aussel22-data"


def norm(signal):
    return (signal - signal.mean()) / np.abs(signal).max()


def band_powers(lfp, out_samp_rate=1000, fname=None):
    """returns power over time (Tx6 array)"""
    band_names = ["theta", "beta2", "gamma", "SWR"]
    band_lim1 = [4, 23, 30, 150]
    band_lim2 = [12, 30, 100, 250]
    f, t, Sxx = signal.spectrogram(lfp, 1024, nperseg=None)
    print(Sxx.shape)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.pcolormesh(t, f, Sxx, cmap="magma")
    # plt.ylim(0, 250)
    for i_band in range(len(band_names))[:1]:
        f_i_band = np.logical_and(f > band_lim1[i_band], f < band_lim2[i_band])
        Sxx_for_band = Sxx[f_i_band, :].sum(axis=0)
        # Sxx_for_band_rel = Sxx_for_band / Sxx[~f_i_band, :].sum(axis=0)
        # Sxx_for_band_rel = Sxx_for_band / Sxx.sum(axis=0)
        # Sxx_for_band_norm = norm(Sxx_for_band)
        Sxx_for_band_norm = Sxx_for_band / Sxx_for_band.max()
        # Sxx_for_band_rel_norm = norm(Sxx_for_band_rel)
        # Sxx_for_band_rel_norm = Sxx_for_band_rel / Sxx_for_band_rel.max()
        ax2.plot(t, Sxx_for_band_norm + i_band, label=band_names[i_band])
    ax2.legend()


for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    fig, ax = plt.subplots()
    if not os.path.isfile(file_path):
        continue
    try:
        data = np.loadtxt(file_path)
    except ValueError:
        # in case of printed list instead of numpy format
        # don't use np.loadtxt
        with open(file_path, "r") as f:
            data = f.read()
        data = np.array(ast.literal_eval(data))
    ax.plot(data)
    ax.set(title=filename)
    # band_powers(data)

plt.show()


# %%
def theta_power(lfp):
    theta_llim = 4
    theta_ulim = 10
    nperseg = 2048
    noverlap = nperseg * 7 // 8
    f, t, Sxx = signal.spectrogram(lfp, 1024, nperseg=nperseg, noverlap=noverlap)
    f_i_band = np.logical_and(f >= theta_llim, f < theta_ulim)
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
from aussel_model.aussel_model.model.apply_input import get_FR

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
for i, ax in enumerate(axs):
    filename = f"input_epi_wake_{i+1}.txt"
    file_path = os.path.join(data_dir, filename)
    inputs = np.loadtxt(file_path)

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
inputs = np.load("../results_2024-01-18 14:30:02.512051/input.npz")
fig, ax = plt.subplots()
ax.plot(inputs['inputs1'])
inputs.keys()
ax.plot(np.loadtxt('./aussel22-data/input_epi_wake_1.txt'))