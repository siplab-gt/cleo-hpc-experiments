import argparse

def main(args):
    # TODO(pmin): Implement a neural net here.
    print(args.model)  # Prints the model type.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a neural net")

    parser.add_argument("--model", required=True, help="Model type (resnet or alexnet)")
    parser.add_argument("--niter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--in_dir", required=True, help="Input directory with images")
    parser.add_argument("--out_dir", required=True, help="Output directory with trained model")

    args = parser.parse_args()
    main(args)