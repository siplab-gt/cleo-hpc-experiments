# doesn't work: cleosim really does require >=3.7 because of type annotations
# conda env create -f aussel_model/environment_hipp_sim.yaml
# conda activate hipp_sim
# pip install cleosim==0.1.1 --ignore-requires-python --no-deps

conda create -n hipp2 python=3.7
conda activate hipp2
pip install cleosim