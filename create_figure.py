# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
# %%
fig = plt.figure(constrained_layout=False, figsize=(3, 4))
subfigs = fig.subfigures(3, 1)
# fig, ax = plt.subfigures(6, 1, sharex=True, figsize=(3, 4))
axs = []
for i, (folder, title) in enumerate([
        ('olconst_results', 'Open-loop (pulse)'),
        ('olmodel_results', 'Open-loop (model-based)'),
        ('cl_results', 'Feedback control')
]):
    path = Path(folder)
    subfig = subfigs[i]
    (ax1, ax2) = subfig.subplots(2, 1, sharex=True)
    axs.append((ax1, ax2))
    ax1.set(title=title)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params('x', direction='inout')
    ax2.tick_params('x', bottom=False, labelbottom=False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # inputs
    in_npz = np.load(path / 'input.npz')
    if folder == 'orig_results':
        t, inputs, in_name,  = in_npz['t_s']*1000, in_npz['inputs1'], "$I_{ext}$ (nA)"
        in_color = '#36827F'
        in_color = 'black'
    else:
        t, inputs, in_name,  = in_npz['t_opto_ms'], in_npz['Irr0_mW_per_mm2'],  "$Irr_0$\n(mW/mm$^2$)"
        in_color = '#72b5f2'
    ax1.step(t, inputs, c=in_color, label=in_name)
    ax1.set(ylabel=in_name)

    # TKLFP
    ref = np.load(path / 'ref.npy')
    t_ms = np.load(path / 't_ms_tklfp.npy')
    tklfp = np.load(path / 'tklfp.npy')
    ax2.plot(t_ms, ref, c='lightgray', label='reference')
    ax2.plot(t_ms, tklfp, c='k', label='measured')
    ax2.set(ylabel='TKLFP\n(μV)')

axs[0][1].legend(loc='lower right')

ax2.tick_params('x', bottom=True, labelbottom=True)
ax2.set(xlabel='Time (ms)')
fig.savefig('results/sim-results.svg')

# %%
fig, ax = plt.subplots()
ax.tick_params('x', top=True, labeltop=True)