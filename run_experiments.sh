#!/bin/bash

# ======================================================= #
# The experiment is to try to evoke ripples from 100 to 300
# ms. The original model can do this with endogenous inputs.
# We will use that ripple as the target for closed-loop 
# control. Then we'll use the average closed-loop power as
# a constant level for open-loop.
# ======================================================= #

# original
#                 --f1=5 yields 200-ms pulse
python run_sim.py --f1=5 --runtime=0.4

# fit
python run_sim.py --mode=fit --runtime=0.1