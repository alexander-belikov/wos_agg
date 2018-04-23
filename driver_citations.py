import argparse
import logging
from wos_agg.aux import main_citations, log_levels, main_merge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sourcepath',
                        default='.',
                        help='Path to data file')

    parser.add_argument('-d', '--destpath', default='.',
                        help='Folder to write data to, Default is current folder')

    parser.add_argument('-v', '--verbosity',
                        default='ERROR',
                        help='set level of verbosity, DEBUG, INFO, WARNING, ERROR')

    parser.add_argument('-l', '--logfile',
                        default='./wos_parser.log',
                        help='Logfile path. Defaults to ./wos_parser.log')

    parser.add_argument('-m', '--mode',
                        default='cite',
                        help='mode can be a) cite, b) merge')

    args = parser.parse_args()

    logging.basicConfig(filename=args.logfile, level=log_levels[args.verbosity],
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    if args.mode == 'cite':
        main_citations(args.sourcepath, args.destpath)
    elif args.mode == 'merge':
        main_merge(args.sourcepath, args.destpath)
    else:
        logging.info('exiting driver_citations flow without action ...')

