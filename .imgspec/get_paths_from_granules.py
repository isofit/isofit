#! /usr/bin/python

import argparse
import glob
import os
import sys
import tarfile


def parse_args():
    parser = argparse.ArgumentParser()
    products = ["rdn", "obs_ort", "loc", "igm", "glt"]
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

    dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    instrument = "PRISMA" if dirs[0][:3] == "PRS" else "AVIRIS"

    # Get paths based on product type file matching
    paths = []
    if instrument == "PRISMA":
        if args.product == "rdn":
            paths = glob.glob(os.path.join(input_dir, "*", "*rdn_prj"))
        elif args.product == "obs_ort":
            paths = glob.glob(os.path.join(input_dir, "*", "*obs_prj"))
        elif args.product == "loc":
            paths = glob.glob(os.path.join(input_dir, "*", "*loc_prj"))
    elif instrument == "AVIRIS":
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
    print(",".join(paths))


if __name__ == "__main__":
    main()
