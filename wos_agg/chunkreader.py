from os import listdir
import pickle
import gzip
from os.path import isfile, join
from gc import collect


class ChunkReader:

    def __init__(self, fpath, prefix, suffix, globar_year):
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
        print(len(last_batch), len(first_batch))

        first_batch.extend(last_batch)
        # queue-like usage
        self.files = first_batch[::-1]

    def info(self):
        print(self.files)

    def empty(self):
        return bool(self.files)

    def pop(self):
        collect()
        if self.files:
            f = self.files.pop()
            with gzip.open(join(self.fpath, f)) as fp:
                item = pickle.load(fp)
                return item
