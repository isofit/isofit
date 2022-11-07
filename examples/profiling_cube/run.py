import argparse

from isofit.core.isofit import Isofit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    args  = parser.parse_args()

    model = Isofit(args.config)
    model.run()
