# [Cleo](https://cleosim.rtfd.io) paper virtual experiment 3
*evoking sharp wave-ripples with optogenetics*

This project takes the hippocampus model developed by Aussel et al. ([2018](https://link.springer.com/article/10.1007/s10827-018-0704-x), [2022](https://link.springer.com/epdf/10.1007/s10827-022-00829-5?sharing_token=Sw7RTIkQRaLgVO28K0KxNfe4RwlQNchNByi7wbcMAY7ikgvvZg602Tl3ZqpP40WLdqEJ2UxRZTBw0DOwGRH380A4Arj7YNkHR4M-sekgxxe7hOLNqxYR4Mo_zCbX_90PhEWk4ggVPRK-gbSfz4PGmOSwPO3auonOH3sXPFWmiG0%3D), [code](https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=266796&file=/model_hipp_final/model_files/#tabs-1)), wraps it with [Cleo](https://cleosim.rtfd.io), and evokes SWRs with optogenetics rather than the original external current meant to model slow-wave sleep rhythms.

## Installation

Just use conda:
```bash
conda create -f environment.yml
conda activate hipp2
```
This creates an environment called `hipp2`&mdash;so called because the original simulation provided an environment called `hipp`.

## Running experiments

Then, follow along with the experiments in `run_experiments.sh`. 
I haven't tested running that file all the way&mdash;rather, I added lines as I worked on the terminal. 
So, I'd recommend running line by line, especially since the simulations take a while (those with full optogenetics take about 10 minutes on my decent laptop).
With a future version of Cleo they should run faster, with more efficient multi-light-source optogenetics.

## Plotting

Some plots are generated automatically by the simulation. That code is in `plot_results.py`. The code I used to make the final summary figure is in `create_figure.py`. 