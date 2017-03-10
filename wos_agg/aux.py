import pickle
import logging
import gzip
from shutil import copyfileobj
from os import listdir
from os.path import isfile, join
from pandas import DataFrame
from numpy import vstack
from .chunkreader import ChunkReader
from .accumulator import Accumulator, AccumulatorOrgs
from graph_tools.ef import calc_eigen_vec

log_levels = {
    "DEBUG": logging.DEBUG, "INFO": logging.INFO,
    "WARNING": logging.WARNING, "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
    }


def is_int(x):
    try:
        int(x)
    except:
        return False
    return True


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
                           (f[-suffix_len:] == suffix and f[:prefix_len] == prefix)]
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

    :param refs: list of references
    :param year: int, current year
    :param delta: int, length of lookback window
    :param filter_wos: bool
    :return:
    """
    if refs:
        filtered = refs
        if filter_wos:
            filtered = filter(lambda x: not filter_wos or x['uid'].startswith('WOS:'), refs)
        # keep the ref if 'year is not available
        # or if delta is not provided or if it is provided
        # and year is greater then current year - delta
        filtered = filter(lambda x: ('year' not in x.keys() or
                                     not is_int(x['year']) or not delta or x['year'] > year - delta - 1), filtered)
        # only keep uids
        filtered = map(lambda x: x['uid'], filtered)
    else:
        filtered = []
    return filtered


def pdata2citations(pdata, delta=None, keep_issn=True, filter_wos=True):
    """
    pdata : list of publication info dicts
    returns cdata: list of citation data tuples
    cdata : [wA, [wBs]]
    """
    pdata_journals = filter(is_article, pdata)
    cdata = []
    for p in pdata_journals:
        refs_ = p['references']
        refs = list(soft_filter_year(refs_, p['date']['year'], delta, filter_wos))
        if keep_issn:
            item = p['id'], p['properties']['issn_int'], refs
        else:
            item = p['id'], refs
        if refs:
            cdata.append(item)
    return cdata


def pub2article_journal(pdata):

    pdata_journals = filter(is_article, pdata)
    aj_data = list(map(lambda x: (x['id'], x['properties']['issn_int']),
                   pdata_journals))
    return aj_data


def gunzip_file(fname_in, fname_out):
    with gzip.open(fname_in, 'rb') as f_in:
        with open(fname_out, 'wb') as f_out:
            copyfileobj(f_in, f_out)


def main(sourcepath, destpath, global_year):
    cr = ChunkReader(sourcepath, 'good', 'pgz', global_year)
    ac = Accumulator(id_type_str=True, prop_type_str=False)
    ac_org = AccumulatorOrgs()
    logging.info(' : global year {0}'.format(global_year))
    raw_refs = 0
    filtered_refs = 0
    while cr.not_empty():
        batch = cr.pop()
        # implicit assumption : all record have the same year within the batch
        batch_year = batch[0]['date']['year']
        logging.info(' : batch year {0}'.format(batch_year))
        aj = pub2article_journal(batch)
        logging.info(' : aj len {0}'.format(len(aj)))
        ac.process_id_prop_list(aj, batch_year != global_year)

        if batch_year == global_year:
            raw_refs_len = sum(map(lambda x: len(x['references']), batch))
            cite_data = pdata2citations(batch, delta=5, keep_issn=False)
            logging.info(' : cite_data len {0}'.format(len(cite_data)))
            filtered_refs_len = sum(map(lambda x: len(x[1]), cite_data))
            logging.info(' : cite_data len of raw refs {0}'.format(raw_refs_len))
            logging.info(' : cite_data len of filtered refs {0}'.format(filtered_refs_len))
            ac.process_id_ids_list(cite_data)
            raw_refs += raw_refs_len
            filtered_refs_len += filtered_refs

        flat_list = ac_org.process_acc(batch)
        ac_org.update(flat_list)
        ac.info()

        logging.info(' main() : cite_data len of raw refs {0}'.format(raw_refs))
        logging.info(' main() : cite_data len of filtered refs {0}'.format(filtered_refs))

    zij, freq, index = ac.retrieve_zij_counts_index()
    logging.info(' main() : citation matrix retrieved')
    ef, ai = calc_eigen_vec(zij, freq, alpha=0.85, eps=1e-6)
    logging.info(' main() : eigenfactor computed')
    df_out = DataFrame(data=vstack([index, ef, ai]).T, columns=['issns', 'ef', 'ai'])
    df_out.to_csv(join(destpath, 'ef_ai_{0}.csv.gz'.format(global_year)), compression='gzip')
    ac_org.dump(join(destpath, 'orgs_{0}.pgz'.format(global_year)))
