"""
Implementation of Baselines
"""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=None, 
                        choices=[None, "Mixup", "RZS", "both"],
                        help='What kind of data augmentation.')
    parser.add_argument('-m', default="Shallow", 
                        choices=["Shallow", "Deep"],
                        help='Which model.')
    args = parser.parse_args()
    print(args.d == None)
    print(args.m)
