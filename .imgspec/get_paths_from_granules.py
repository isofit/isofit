#! /usr/bin/python

import argparse
import glob
import os
import sys
import tarfile


def parse_args():
    parser = argparse.ArgumentParser()
    products = ["rdn", "obs_ort", "loc", "igm", "glt", "rfl", "topo", "brdf", "topo_brdf"]
    formats = ["envi", "hdf"]
    parser.add_argument("-p", "--product",
                        help=("Choose one of the following product types: " + ", ".join(products)))
    parser.add_argument("-f", "--format",
                        help=("Choose one of the following formats: " + ", ".join(formats)))
    args = parser.parse_args()

    if args.product:
        if args.product not in products:
            print("ERROR: Product \"%s\" is not a valid product choice." % args.product)
            sys.exit(1)
    if args.format:
        if args.format not in formats:
            print("ERROR: Format \"%s\" is not a valid format choice." % f)
            sys.exit(1)
    return args


def main():
    args = parse_args()

    # Unzip and untar granules
    input_dir = "input"
    granule_paths = glob.glob(os.path.join(input_dir, "*.tar.gz"))
    for g in granule_paths:
        tar_file = tarfile.open(g)
        tar_file.extractall(input_dir)
        tar_file.close()
        os.remove(g)

    # Get paths based on product type file matching
    if args.product == "rdn":
        paths = glob.glob(os.path.join(input_dir, "*rdn*", "*rdn*img"))
    elif args.product == "obs_ort":
        paths = glob.glob(os.path.join(input_dir, "*rdn*", "*obs_ort"))
    elif args.product == "loc":
        paths = glob.glob(os.path.join(input_dir, "*rdn*", "*loc"))
    elif args.product == "igm":
        paths = glob.glob(os.path.join(input_dir, "*rdn*", "*igm"))
    elif args.product == "glt":
        paths = glob.glob(os.path.join(input_dir, "*rdn*", "*glt"))
    elif args.product == "rfl":
        paths = glob.glob(os.path.join(input_dir, "*rfl*", "*rfl*img"))
        paths += glob.glob(os.path.join(input_dir, "*refl*", "*rfl*img"))
        paths += glob.glob(os.path.join(input_dir, "*rfl*", "*corr*img"))
        paths += glob.glob(os.path.join(input_dir, "*refl*", "*corr*img"))
    print(",".join(paths))


if __name__ == "__main__":
    main()
