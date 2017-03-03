import argparse
import logging
from wos_agg.aux import main, log_levels, is_int

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sourcepath',
                        default='.',
                        help='Path to data file')

    parser.add_argument('-d', '--destpath', default='.',
                        help='Folder to write data to, Default is current folder')

    parser.add_argument('-y', '--year', default='1985',
                        help='Global year setting')

    parser.add_argument('-v', '--verbosity',
                        default='ERROR',
                        help='set level of verbosity, DEBUG, INFO, WARNING, ERROR')

    parser.add_argument('-l', '--logfile',
                        default='./wos_parser.log',
                        help='Logfile path. Defaults to ./wos_parser.log')

    args = parser.parse_args()

    if is_int(args.maxchunks):
        maxchunks = int(args.maxchunks)
    else:
        maxchunks = None

    logging.basicConfig(filename=args.logfile, level=log_levels[args.verbosity],
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    main(args.sourcepath, args.destpath, args.year)
