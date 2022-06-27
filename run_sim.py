import argparse

from brian2 import Network, mm

import cleosim
from cleosim.electrodes import Probe, TKLFPSignal

def main(args):
    if args.smoke:
        smoke_test()
        return

    net = setup_aussel_net(args)
    sim = cleosim.CLSimulator(net)
    lfp = TKLFPSignal("lfp", save_history=True)
    probe = Probe("probe", [[-1, -1, -1]]*mm, lfp)

def smoke_test():
    from aussel_model import user_interface_simple as uis
    uis.maxN.set(500)
    uis.runtime.set(0.01)
    uis.start()
    uis.save_plots()

def setup_aussel_net(args) -> Network:
    from aussel_model import user_interface_simple as uis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CLEOSim case study 3")

    # smoke test
    # subparsers = parser.add_subparsers(help="sub-commands")
    # parser_smoke = subparsers.add_parser('smoke', help='run a short smoke test')

    # main args
    parser.add_argument("--smoke", action="store_true", default=False, help="Run a short smoke test (set maxN to 500 and runtime to 0.01 s)")
    parser.add_argument("--maxN", type=int, default=10000, help="Choose the maximum number of neurons in the network (in the CA1 excitatory neurons group) :\nThe total number of neurons will be 3.32*N")
    parser.add_argument("--runtime", type=float, default=0.5, help="Duration of the simulation (s)")

    # parser.add_argument("--model", required=True, help="Model type (resnet or alexnet)")
    # parser.add_argument("--niter", type=int, default=1000, help="Number of iterations")
    # parser.add_argument("--in_dir", required=True, help="Input directory with images")
    # parser.add_argument("--out_dir", required=True, help="Output directory with trained model")

    args = parser.parse_args()
    main(args)