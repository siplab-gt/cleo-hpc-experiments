#!/bin/bash

# =========================================================
# The experiment is to try to evoke ripples from 100 to 300
# ms. The original model can do this with endogenous inputs.
# We will use that ripple as the target for closed-loop 
# control after fitting a model to training data.
# =========================================================

function get_last_dir () {
    echo "$(ls -tp | grep /$ | head -1)"
}

# check that aussel_model is working
python run_sim.py --smoke --show_plots

# to get 3D figure
python run_sim.py --runtime=0 --show_plots --mode=OLconst --opto_slice

# original
#                 --f1=5 yields 200-ms pulse
python run_sim.py --f1=5 --runtime=0.4 --mode=orig --target=cython
ln -s "$(get_last_dir)" orig_results

# naïve open-loop
# --ref determines shape of stimulus
python run_sim.py --mode=OLnaive --ref=orig_results/tklfp.npy --n_trials=10 --target=cython --noise
ln -s "$(get_last_dir)" olnaive_results

# fit
#                    I just happened to use 13 seconds of training data
python run_sim.py --mode=fit --runtime=13 --target=cython --noise
ln -s "$(get_last_dir)" fit_results
python fit_data.py fit_results --out=fit_results/fit.npz --iterEM=1000

# LQR
python run_sim.py --mode=LQR --fit=fit_results/fit.npz --ref=orig_results/tklfp.npy --n_trials=10 --target=cython --noise
ln -s "$(get_last_dir)" lqr_results

# MPC
python run_sim.py --mode=MPC --fit=fit_results/fit.npz --ref=orig_results/tklfp.npy --n_trials=10 --target=cython --noise
ln -s "$(get_last_dir)" mpc_results

# ======================================================================
# Validation, reproducing epilepsy figure from Aussel 2022 figure 5
# ======================================================================
python run_sim.py --mode=val-epi --runtime=35 --target=cython
ln -s "$(get_last_dir)" val_epi_results

python run_sim.py --mode=val-healthy --runtime=35 --target=cython
ln -s "$(get_last_dir)" val_healthy_results