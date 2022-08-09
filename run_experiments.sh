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
python run_sim.py --f1=5 --runtime=0.4

# fit
python run_sim.py --mode=fit --runtime=13 --maxN=10000 --maxIrr0=30 --target=cython
# most recent results
FIT_RESULTS = $(get_last_dir)
mkdir -p results
python fit_data.py $FIT_RESULTS --out=results/fit.npz --iterEM=1000

# closed-loop