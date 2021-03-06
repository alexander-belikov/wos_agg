from os import listdir
import pickle
import gzip
from os.path import isfile, join
from gc import collect
import logging


class ChunkReader(object):

    def __init__(self, fpath, prefix, suffix, globar_year=None):
        """
            read files from fpath with files with globar_year read last

        :param fpath: filepath
        :param prefix: prefix of files
        :param suffix: suffix of files
        :param globar_year: year specifier of files
        """
        self.fpath = fpath
        prefix_len = len(prefix)
        suffix_len = len(suffix)
        self.year = globar_year
        self.year_str = str(globar_year)

        # filename example : good_2005_0.pgz
        files = [f for f in listdir(fpath) if isfile(join(fpath, f)) and
                 (f[-suffix_len:] == suffix and f[:prefix_len] == prefix)]

        pat = '{0}_{1}'.format(prefix, self.year_str)
        last_batch = \
            list(sorted(filter(lambda f: pat in f, files)))
        first_batch = \
            list(sorted(filter(lambda f: pat not in f, files)))

        logging.info('in ChunkReader.__init__() : '
                     '(last_batch) {0} files : {1}'.format(len(last_batch), ' '.join(last_batch)))
        logging.info('in ChunkReader.__init__() : '
                     '(first_batch) {0} files : {1}'.format(len(first_batch), ' '.join(first_batch)))

        first_batch.extend(last_batch)
        # queue-like usage
        self.files = first_batch[::-1]
        logging.info('in ChunkReader.__init__ : '
                     'all files {0} files : {1}'.format(len(self.files), ' '.join(self.files)))

    def info(self):
        logging.info('{0}'.format(' '.join(self.files)))

    def not_empty(self):
        return bool(self.files)

    def pop(self):
        """
        pop the current item
        :return:
        """
        collect()
        if self.files:
            f = self.files.pop()
            logging.info(' in ChunkReader.pop() : trying to open file {0}'.format(f))
            with gzip.open(join(self.fpath, f)) as fp:
                item = pickle.load(fp)
                logging.info(' in ChunkReader.pop() : file {0} loaded'.format(f))
                return item
