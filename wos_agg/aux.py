import pickle
import logging
import gzip
from shutil import copyfileobj
from os import listdir
from os.path import isfile, join
from pandas import DataFrame
from numpy import vstack
from .chunkreader import ChunkReader
from .accumulator import Accumulator, AccumulatorOrgs, AccumulatorCite
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


def soft_filter_year(refs, year, delta=None, filter_wos=True, keep_year=False):
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
        # keep the ref if year is not available
        # or if delta is not provided or if it is provided
        # and year is greater then current year - delta
        if delta:
            filtered = filter(lambda x: ('year' not in x.keys() or
                                         not is_int(x['year']) or x['year'] > year - delta - 1), filtered)
        # only keep uids
        if keep_year:
            filtered = map(lambda x: (x['uid'], None if 'year' not in x.keys() else x['year']), filtered)
        else:
            filtered = map(lambda x: x['uid'], filtered)
    else:
        filtered = []
    return filtered


def pdata2citations(pdata, delta=None, keep_issn=True, keep_year=False, filter_wos=True, filter_articles=True):
    """
    pdata : list of publication info dicts
    returns cdata: list of citation data tuples
    cdata : [wA, [wBs]]
    """
    if filter_articles:
        pdata2 = filter(is_article, pdata)
    else:
        pdata2 = pdata

    cdata = []
    for p in pdata2:
        refs_ = p['references']
        refs = list(soft_filter_year(refs_, p['date']['year'], delta, filter_wos, keep_year))
        if keep_issn:
            item = p['id'], p['properties']['issn_int'], refs
        else:
            item = p['id'], refs
        cdata.append(item)
    return cdata


def pub2article_journal(pdata):

    pdata_journals = filter(is_article, pdata)
    aj_data = list(map(lambda x: (x['id'], x['properties']['issn_int']),
                   pdata_journals))
    return aj_data


def pub2issn_title(pdata):

    pdata_journals = filter(is_article, pdata)
    pdata_journals = filter(lambda x: 'properties' in x.keys(), pdata_journals)
    pdata_journals = filter(lambda x: 'source_title' in x['properties'].keys(), pdata_journals)
    jt_data = list(map(lambda x: (x['properties']['issn_int'], x['properties']['source_title']), pdata_journals))
    return jt_data


def gunzip_file(fname_in, fname_out):
    with gzip.open(fname_in, 'rb') as f_in:
        with open(fname_out, 'wb') as f_out:
            copyfileobj(f_in, f_out)


def main(sourcepath, destpath, global_year, max_list_len):
    cr = ChunkReader(sourcepath, 'good', 'pgz', global_year)
    ac = Accumulator(id_type_str=True, prop_type_str=False, max_list_len=max_list_len)
    ac_org = AccumulatorOrgs()
    jt_dict = dict()
    logging.info(' : global year {0}'.format(global_year))
    raw_refs = 0
    filtered_refs = 0
    while cr.not_empty():
        batch = cr.pop()
        # implicit assumption : all record have the same year within the batch
        batch_year = batch[0]['date']['year']
        logging.info(' main() : batch year {0}'.format(batch_year))
        aj = pub2article_journal(batch)
        logging.info(' main() : aj len {0}'.format(len(aj)))
        ac.process_id_prop_list(aj, batch_year != global_year)

        jt = pub2issn_title(batch)
        jt_dict.update(jt)

        if batch_year == global_year:
            raw_refs_len = sum(map(lambda x: len(x['references']), batch))
            cite_data = pdata2citations(batch, delta=5, keep_issn=False)
            logging.info(' main() : cite_data len {0}'.format(len(cite_data)))
            filtered_refs_len = sum(map(lambda x: len(x[1]), cite_data))
            logging.info(' main() : cite_data len of raw refs {0}'.format(raw_refs_len))
            logging.info(' main() : cite_data len of filtered refs {0}'.format(filtered_refs_len))
            ac.process_id_ids_list(cite_data)
            raw_refs += raw_refs_len
            filtered_refs += filtered_refs_len
            # accumulate organization data
            ac_org.process_acc(batch)
        ac.info()

        logging.info(' main() : total raw refs {0}'.format(raw_refs))
        logging.info(' main() : total filtered refs {0}'.format(filtered_refs))

    zij, freq, index = ac.retrieve_zij_counts_index()
    logging.info(' main() : citation matrix retrieved')
    ef, ai = calc_eigen_vec(zij, freq, alpha=0.85, eps=1e-6)
    logging.info(' main() : eigenfactor computed')
    df_out = DataFrame(data=vstack([index, ef, ai]).T, columns=['issn', 'ef', 'ai'])
    df_out.to_csv(join(destpath, 'ef_ai_{0}.csv.gz'.format(global_year)), compression='gzip')
    ac_org.dump(join(destpath, 'affs_{0}.pgz'.format(global_year)))
    with gzip.open(join(destpath, 'jt_{0}.pgz'.format(global_year)), 'wb') as fp:
        pickle.dump(jt_dict, fp)


def main_citations(sourcepath, destpath):
    """
    WIP
    :param sourcepath:
    :param destpath:
    :return:
    """
    cr = ChunkReader(sourcepath, 'good', 'pgz')
    ag = AccumulatorCite()
    raw_refs = 0
    filtered_refs = 0

    while cr.not_empty():
        batch = cr.pop()

        cite_data = pdata2citations(batch, delta=None, keep_issn=False, filter_articles=False, keep_year=True)
        year_data = [None if 'year' not in x['date'].keys() else x['date']['year'] for x in batch]
        month_data = [None if 'month' not in x['date'].keys() else x['date']['month'] for x in batch]
        day_data = [None if 'day' not in x['date'].keys() else x['date']['day'] for x in batch]
        id_data = [x['id'] for x in batch]

        ref_id_year_ss = [item[1] for item in cite_data]
        ref_ids = [ref_id for sublist in ref_id_year_ss for ref_id, y in sublist]
        ag.update_set_map(id_data + ref_ids)

        ag.update_dates(id_data, zip(year_data, month_data, day_data), False)

        # lists of refs (id_str, y)
        ref_id_year_ss = [item[1] for item in cite_data]
        # flat list of refs
        ref_id_year = [item for sublist in ref_id_year_ss for item in sublist]
        ref_id_data = [x for x, y in ref_id_year]
        ref_date_data = [(y, None, None) for x, y in ref_id_year]

        ag.update_dates(ref_id_data, ref_date_data, False, False)

        ag.update_citations(cite_data, False)

        raw_refs_len = sum(map(lambda x: len(x['references']), batch))

        logging.info(' main() : cite_data len {0}'.format(len(cite_data)))
        filtered_refs_len = sum(map(lambda x: len(x[1]), cite_data))
        logging.info(' main() : cite_data len of raw refs {0}'.format(raw_refs_len))
        logging.info(' main() : cite_data len of filtered refs {0}'.format(filtered_refs_len))
        raw_refs += raw_refs_len
        filtered_refs += filtered_refs_len

    ag.dump(join(destpath, 'cite_pack.pgz'))
