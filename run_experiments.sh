#!/bin/bash

# ======================================================= #
# The experiment is to try to evoke ripples from 100 to 300
# ms. The original model can do this with endogenous inputs.
# We will use that ripple as the target for closed-loop 
# control. Then we'll use the average closed-loop power as
# a constant level for open-loop.
# ======================================================= #

function get_last_dir () {
    echo "$(ls -tp | grep /$ | head -1)"
}

# original
#                 --f1=5 yields 200-ms pulse
python run_sim.py --f1=5 --runtime=0.4 --mode=orig --target=cython
ln -s $(get_last_dir) orig_results

# fit
#                    I just happened to use 13 seconds of training data
python run_sim.py --mode=fit --runtime=13 --target=cython
ln -s $(get_last_dir) fit_results
python fit_data.py fit_results --out=fit_results/fit.npz --iterEM=1000

# closed-loop
python run_sim.py --mode=CL --fit=fit_results/fit.npz --ref=orig_results/tklfp.npy --runtime=0.4 --target=cython
ln -s $(get_last_dir) cl_results

# open-loop constant input
IRR0="$(python get_OLconst_level.py cl_results/input.npz)"
# --ref passed just for plotting
python run_sim.py --mode=OLconst --Irr0_OL=${IRR0} --ref=orig_results/tklfp.npy --runtime=0.4 --target=cython
ln -s $(get_last_dir) olconst_results

# open-loop model-based
python run_sim.py --mode=OLmodel --fit=fit_results/fit.npz --ref=orig_results/tklfp.npy --runtime=0.4 --target=cython
ln -s $(get_last_dir) olmodel_results


# ======================
# REDO, WITH NOISE ADDED
# ======================
# fit
#                    I just happened to use 13 seconds of training data
python run_sim.py --mode=fit --runtime=13 --target=cython --noise
ln -s $(get_last_dir) fit_results_noise
python fit_data.py fit_results --out=fit_results_noise/fit.npz --iterEM=1000

# closed-loop
python run_sim.py --mode=CL --fit=fit_results_noise/fit.npz --ref=orig_results/tklfp.npy --n_trials=10 --target=cython --noise
ln -s $(get_last_dir) cl_results_noise

# NAIVE INSTEAD OF SQUARE PULSE
# --->>-----------------<<-----
# --ref determines shape of stimulus
python run_sim.py --mode=OLnaive --ref=orig_results/tklfp.npy --n_trials=10 --target=cython --noise
ln -s $(get_last_dir) olnaive_results_noise

# open-loop model-based
python run_sim.py --mode=OLmodel --fit=fit_results_noise/fit.npz --ref=orig_results/tklfp.npy --n_trials=10 --target=cython --noise
ln -s $(get_last_dir) olmodel_results_noise

