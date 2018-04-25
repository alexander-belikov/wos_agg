import pickle
import logging
import gzip
from pympler.asizeof import asizeof
from shutil import copyfileobj
from os import listdir, rename, sysconf
from os.path import isfile, join
from pandas import DataFrame
from numpy import vstack
from .chunkreader import ChunkReader
from .accumulator import Accumulator, AccumulatorOrgs, AccumulatorCite
from graph_tools.ef import calc_eigen_vec
import pathos.multiprocessing as mp
from re import findall
from functools import partial
import gc

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
        ag.update_with_cite_data(cite_data)

        id_data = [x['id'] for x in batch]
        date_data = [x['date'] for x in batch]
        ag.update_with_pub_data(id_data, date_data)

        raw_refs_len = sum(map(lambda x: len(x['references']), batch))

        logging.info(' main_citations() : cite_data len {0}'.format(len(cite_data)))
        filtered_refs_len = sum(map(lambda x: len(x[1]), cite_data))
        logging.info(' main_citations() : cite_data len of raw refs {0}'.format(raw_refs_len))
        logging.info(' main_citations() : cite_data len of filtered refs {0}'.format(filtered_refs_len))
        raw_refs += raw_refs_len
        filtered_refs += filtered_refs_len

    ag.dump(join(destpath, 'cite_pack.pgz'))


def main_merge(sourcepath, destpath, n_processes=2):
    """
    :param sourcepath:
    :param destpath:
    :return:
    """
    suffix = 'pgz'
    suffix_len = len(suffix)
    ls = suffix_len
    prefix = 'cite_pack'
    prefix_len = len(prefix)
    fpath = sourcepath
    mem_bytes = sysconf('SC_PAGE_SIZE') * sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
    mem_gib = mem_bytes / 1024 ** 3

    files = sorted([f for f in listdir(fpath) if isfile(join(fpath, f)) and
                    (f[-suffix_len:] == suffix and f[:prefix_len] == prefix)])[::-1]

    files = [(join(fpath, f), 10) for f in files]
    logging.info(' main_acs() : files list: {0}'.format(files))

    while len(files) > 1:
        bnd = min(len(files), 2*n_processes)//2
        fname_merge, fname_untouched = files[:2*bnd], files[2*bnd:]
        logging.info('main_merge() : len acs to merge : {0} leftover len : {1}'.format(len(fname_merge),
                                                                                     len(fname_untouched)))
        fname_pairs = zip(fname_merge[::2], fname_merge[1::2])
        fname_triplet = [(fa, fb,
                         fa[0][:-ls] + findall(r'{0}(.*){1}'.format(prefix, suffix), fb[0])[0].lstrip('.')
                         + suffix) for fa, fb in fname_pairs]
        func = partial(merge_acs)

        expected_mem_occupation = sum([s for _, s in fname_merge])
        n_proc_adj = int(n_processes * mem_gib * 0.8 / expected_mem_occupation)
        n_proc = min(n_processes, n_proc_adj)
        with mp.Pool(n_proc) as p:
            fname_merge = p.map(func, fname_triplet)
        files = fname_merge + fname_untouched
        gc.collect()
    logging.info(' main_merge() : {0} {1}'.format(files[0][0], files[0][1]))
    rename(files[0][0], join(destpath, 'all_cite_pack.pgz'))


def merge_acs(fname_triplet):
    fa_full, fb_full, fnew = fname_triplet
    fa, _ = fa_full
    fb, _ = fb_full
    a = AccumulatorCite(fa)
    b = AccumulatorCite(fb)
    a.load()
    size_a = asizeof(a) / 1024 ** 3
    logging.info(' main_acs() : {0} a size {1:.1f} Gb'.format(a.fname, size_a))
    b.load()
    size_b = asizeof(b) / 1024 ** 3
    logging.info(' main_acs() : {0} b size {1:.1f} Gb'.format(b.fname, size_b))
    a.merge(b)
    size_new = asizeof(a) / 1024 ** 3
    logging.info(' main_acs() : merged and b size {0:.1f} Gb'.format(size_new))
    del b
    a.dump(fnew)
    del a
    gc.collect()
    return fnew, size_new
