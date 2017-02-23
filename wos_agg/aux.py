from os import listdir
from os.path import isfile, join
import gzip
import pickle


def fetch_accumulator(fpath, prefix, suffix):
    """

    :param fpath:
    :param prefix:
    :param suffix:
    :return:
    """
    prefix_len = len(prefix)
    suffix_len = len(suffix)
    prefix_suffix_files = [f for f in listdir(fpath) if isfile(join(fpath, f)) and
                           (f[-suffix_len:] == '.pgz' and f[:prefix_len] == prefix)]
    acc = []
    for f in prefix_suffix_files:
        with gzip.open(join(fpath, f)) as fp:
            item = pickle.load(fp)
            acc.extend(item)
    return acc


def is_article(x):
    return x['properties']['pubtype'] == 'Journal' \
        and 'issn' in x['properties'].keys() and 'issn_int' in x['properties'].keys()


def soft_filter_year(refs, year, delta=None, filter_wos=True):
    """
    filter_wos ? then check startstwith
    """

    f1 = filter(lambda x: not filter_wos or x['uid'].startswith('WOS:'), refs)
    # keep the ref if 'year is not available
    # or if delta is not provided or if it is provided
    # and year is greater then current year - delta
    f2 = filter(lambda x: ('year' not in x.keys() or not delta or x['year'] > year - delta - 1), f1)
    f3 = map(lambda x: x['uid'], f2)
    return f3


def pdata2citations(pdata, delta=None, keep_issn=True):
    """
    pdata : list of publication info dicts
    returns cdata: list of citation data tuples
    cdata : [wA, [wBs]]
    """
    pdata_journals = filter(is_article, pdata)
    cdata = []
    for p in pdata_journals:
        refs = list(soft_filter_year(p['references'], p['date']['year'], delta, True))
        if keep_issn:
            item = p['id'], p['properties']['issn'], refs
        else:
            item = p['id'], refs
        if refs:
            cdata.append(item)
    return cdata