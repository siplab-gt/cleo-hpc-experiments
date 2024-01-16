# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
# make text in SVG editable:
plt.rc('svg', fonttype='none')
# %%
fig = plt.figure(constrained_layout=False, figsize=(4, 4))
subfigs = fig.subfigures(3, 1)
axs = []
for i, (folder, title) in enumerate([
        ('olnaive_results', 'Open-loop (naïve)'),
        ('lqr_results', 'LQR feedback control'),
        ('mpc_results', 'Model-predictive feedback control (MPC)'),
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

    trial_len = 400

    # inputs
    in_npz = np.load(path / 'input.npz')
    if folder == 'orig_results':
        raise NotImplementedError("won't work for multiple trials because of different sampling frequency")
        t, inputs, in_name,  = in_npz['t_s']*1000, in_npz['inputs1'], "$I_{ext}$ (nA)"
        in_color = '#36827F'
        in_color = 'black'
    else:
        t, inputs, in_name,  = in_npz['t_opto_ms'], in_npz['Irr0_mW_per_mm2'],  "Irr$_0$\n(mW/mm$^2$)"
        in_color = '#72b5f2'
    for i_trial in range(max(1, len(t)//trial_len)):
        # need to search for indices bc CL doesn't have exactly 1 sample/ms
        t1 = i_trial * trial_len
        t2 = (i_trial + 1) * trial_len
        i1, i2 = np.searchsorted(t, (t1, t2))
        ax1.plot(t[i1:i2]-t[i1], inputs[i1:i2], c=in_color, lw=1, alpha=.2, label=in_name)
    ax1.set(ylabel=in_name)

    # TKLFP
    ref = np.load(path / 'ref.npy')
    t_ms = np.load(path / 't_ms_tklfp.npy')
    tklfp = np.load(path / 'tklfp.npy')
    (line_ref,) = ax2.plot(t_ms[:trial_len], ref[:trial_len], c='#c500cc', label='reference')
    for i_trial in range(len(t_ms)//trial_len):
        i1 = i_trial * trial_len
        i2 = (i_trial + 1) * trial_len
        (line_meas,) = ax2.plot(t_ms[:trial_len], tklfp[i1:i2], c='k', alpha=.2, lw=1, label='measured')
    ax2.set(ylabel='TKLFP\n(μV)')

axs[0][1].legend(handles=[line_ref, line_meas], loc='lower right')

ax2.tick_params('x', bottom=True, labelbottom=True)
ax2.set(xlabel='Time (ms)')
fig.savefig('results/sim-results.svg', bbox_inches="tight", transparent=True)

# %%
# comparing TKLFP and RWSLFP
import seaborn as sns
from aussel_model.model.single_process3 import lecture
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.5, 4))
rwslfp = -np.array(lecture('orig_results/LFP.txt')).flatten() * 1e6
ax1.plot(np.arange(0, 399, step=1000/1024), rwslfp, c='k')
ax1.set(ylabel='μV', title='Reference weighted sum LFP approximation')
tklfp = np.load('orig_results/tklfp.npy')
ax2.plot(np.arange(400), tklfp, c='k')
ax2.set(ylabel='μV', xlabel='t (ms)', title='Teleńczuk kernel LFP approximation')
sns.despine(fig)
fig.savefig('results/rws_tk_lfp.pdf', transparent=True, bbox_inches='tight')
# %%
