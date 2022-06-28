import argparse
import time

import matplotlib.pyplot as plt
from brian2 import Network, mm, ms

import cleosim
from cleosim.electrodes import Probe, TKLFPSignal

from aussel_model.model_files import single_process3 as sp3
from aussel_model import user_interface_simple as uis

def main(args):
    setup_start = time.time()
    (net, all_ngs, elec_pos), params = setup_aussel_net(args)
    all_ngs_exc = [area[0][0] for area in all_ngs]
    all_ngs_inh = [area[1][0] for area in all_ngs]

    assign_coords(all_ngs)
    sim = cleosim.CLSimulator(net)

    sim.set_io_processor(cleosim.processing.RecordOnlyProcessor(1))

    lfp = TKLFPSignal("lfp", save_history=True)
    # use same electrode coordinates, with the same 150um scale
    probe = Probe("probe", elec_pos*0.15*mm, [lfp])
    sim.inject_recorder(probe, *all_ngs_exc, tklfp_type='exc')
    sim.inject_recorder(probe, *all_ngs_inh, tklfp_type='inh')

    if args.viz:
        cleosim.visualization.plot(*all_ngs_exc, *all_ngs_inh, devices_to_plot=[probe])

    print(f'Setup time: {(time.time()-setup_start)} seconds')

    sp3.run_process(net, all_ngs, elec_pos, *params)

    plt.figure()
    plt.plot(lfp.lfp_uV)
    plt.title("TKLFP")

    uis.aborted = False
    uis.save_plots()
    if args.show_plots:
        plt.show()



def assign_coords(all_ngs):
    # all_ngs is 4 x 2 x 1 nested list.
        # 4 areas, 2 types, and I don't know why the last level:
        # aussel_model/model_files/preparation.py:337
    for i_area in range(4):  # EC, DG, CA3, CA1
        for i_type in range(2):  # exc, inh
            ng = all_ngs[i_area][i_type][0]
            cleosim.coordinates.assign_coords(ng, ng.x_soma/mm, ng.y_soma/mm, ng.z_soma/mm)



def setup_aussel_net(args) -> Network:
    uis.maxN.set(args.maxN)
    uis.runtime.set(args.runtime)
    if args.smoke:
        uis.maxN.set(500)
        uis.runtime.set(0.01)
    if args.save_neuron_pos:
        uis.save_neuron_pos.set('True')

    params = uis.get_process_params()
    return sp3.net_setup(*params, plot_topo=args.plot_topo), params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CLEOSim case study 3")

    parser.add_argument("--smoke", action="store_true", default=False, help="Run a short smoke test (set maxN to 500 and runtime to 0.01 s)")

    # args from original interface
    parser.add_argument("--maxN", type=int, default=10000, help="Choose the maximum number of neurons in the network (in the CA1 excitatory neurons group) :\nThe total number of neurons will be 3.32*N")
    parser.add_argument("--runtime", type=float, default=0.5, help="Duration of the simulation (s)")
    parser.add_argument("--save_neuron_pos", action="store_true", default=False, help="Save neuron positions as txt files")
    
    # visualization
    parser.add_argument("--plot_topo", action="store_true", default=False, help="Plot neuron and electrode positions")
    parser.add_argument("--viz", action="store_true", default=False, help="Visualize using CLEOSim utilities")
    parser.add_argument("--show_plots", action="store_true", default=False, help="Show interactive plot windows after simulation")

    # args for wrapping with CLEOSim 

    # parser.add_argument("--model", required=True, help="Model type (resnet or alexnet)")
    # parser.add_argument("--niter", type=int, default=1000, help="Number of iterations")
    # parser.add_argument("--in_dir", required=True, help="Input directory with images")
    # parser.add_argument("--out_dir", required=True, help="Output directory with trained model")

    args = parser.parse_args()
    main(args)