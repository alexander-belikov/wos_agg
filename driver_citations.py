import argparse
import logging
from wos_agg.aux import main_citations, log_levels, main_merge, main_retrieve_cite_data
import sys

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
                        default='STDOUT',
                        help='Logfile path. Defaults to STDOUT')

    parser.add_argument('-m', '--mode',
                        default='cite',
                        help='mode can be a) cite, b) merge')

    parser.add_argument('-n', '--nproc',
                        default=1, type=int,
                        help='number of threads')

    parser.add_argument('-g', '--max-gb-pickle',
                        default=20, type=float,
                        help='max size of intermediate pickle in Gb')

    parser.add_argument('-w', '--widsfname',
                        default='wosids.csv.gz', type=str,
                        help='name of the wosids csv gz file')

    args = parser.parse_args()

    logging.basicConfig(filename=args.logfile, level=log_levels[args.verbosity],
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    log = logging.getLogger()

    if args.logfile == 'STDOUT':
        ch = logging.StreamHandler(sys.stdout)
    else:
        ch = logging.StreamHandler(open(args.logfile, 'w'))

    ch.setLevel(log_levels[args.verbosity])
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if args.mode == 'cite':
        main_citations(args.sourcepath, args.destpath)
    elif args.mode == 'merge':
        main_merge(args.sourcepath, args.destpath, args.nproc, args.max_gb_pickle)
    elif args.mode == 'retrieve':
        main_retrieve_cite_data(args.sourcepath, args.destpath, wids_fname=args.widsfname)
    else:
        logging.info('exiting driver_citations flow without action ...')

