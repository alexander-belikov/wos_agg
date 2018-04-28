from numpy import nan
from pandas import DataFrame, Series
from networkx import Graph, to_pandas_adjacency, from_pandas_adjacency
from graph_tools.reduction import update_edges, describe_graph, project_graph_return_adj
from graph_tools.adj_aux import create_adj_matrix
import logging
from pympler.asizeof import asizeof
from numpy import dot, arange, array
from gc import collect
import gzip
import pickle
import gc

id_type = 'id'
prop_type = 'prop'


def is_bstr(x):
    try:
        x.encode('latin-1')
    except:
        return True
    return False


#TODO introduce obvious inheritance
class Accumulator(object):

    def __init__(self, id_type_str=False, prop_type_str=False, max_list_len=1000):

        self.type = (id_type, prop_type)

        # is_a_string_type flags
        self.type_str = {id_type: id_type_str, prop_type: prop_type_str}
        # set of ids
        self.sets = {id_type: set(), prop_type: set()}

        # x^str -> x^int (x str to x int dict) x ~ id, prop
        self.str_to_int_maps = {id_type: {}, prop_type: {}}

        # x^int -> x^str (x int to x str dict) x ~ id, prop
        self.int_to_str_maps = {id_type: {}, prop_type: {}}

        self.max_cite_len = max_list_len

        # prop^i : counts // keeps tracks of property frequencies
        self.prop_counts = Series()

        # prop^i -> id^i (prop int to id int)
        self.g_prop_to_id = Graph()
        # prop^i(citing) -> prop^i(cited) (prop int to prop int)
        self.g_prop_to_prop = Graph()

    def process_id_prop_list(self, in_list, update_counts_flag=True):
        """

        :param in_list: [(id, prop)]
        :param update_counts_flag:
        :return: None
        """
        new_ids = map(lambda x: x[0], in_list)
        new_props = map(lambda x: x[1], in_list)

        self.update_sets_maps(new_ids, id_type)
        self.update_sets_maps(new_props, prop_type)

        # obtain int-int list if needed
        new_ids = map(lambda x: x[0], in_list)
        new_props = map(lambda x: x[1], in_list)

        if self.type_str[id_type]:
            new_ids = map(lambda x: self.str_to_int_maps[id_type][x], new_ids)

        if self.type_str[prop_type]:
            new_props = map(lambda x: self.str_to_int_maps[prop_type][x], new_props)

        # in_list_transformed = zip(new_ids, new_props)
        # update self.prop_to_id // Graph()
        for id_, prop_ in zip(new_ids, new_props):
            self.g_prop_to_id.add_edge((prop_type, prop_),
                                       (id_type, id_), {'weight': 1.0})

        if update_counts_flag:
            new_props = map(lambda x: x[1], in_list)
            logging.info(' process_id_prop_list() : updating property counts')
            self.update_prop_counts(new_props)

        logging.info(' process_id_prop_list() : g_prop_to_id composition')
        logging.info(describe_graph(self.g_prop_to_id))

    def update_prop_counts(self, props):
        cur_vc = Series(props).value_counts()
        tmp_vc = DataFrame(index=list(set(self.prop_counts.index) | set(cur_vc.index)),
                           columns=['acc', 'new'])
        tmp_vc.sort_index(inplace=True)
        tmp_vc['acc'].update(self.prop_counts)
        tmp_vc['new'].update(cur_vc)
        tmp_vc = tmp_vc.fillna(0.0)
        tmp_vc['acc'] += tmp_vc['new']
        self.prop_counts = tmp_vc['acc'].copy()
        logging.info(' update_prop_counts() : a total of {0} counted'.format(self.prop_counts.sum()))

    def process_id_ids_list(self, in_list):
        delta = self.max_cite_len
        if len(in_list) > delta:
            super_list = [in_list[k:k + delta] for k in arange(0, len(in_list), delta)]
            logging.info(' process_id_ids_list() : len of super list {0}'.format(len(super_list)))
            k = 0
            for item in super_list:
                logging.info('{0} {1}'.format(k, len(item)))
                self.process_id_ids_list_(item)
                collect()
                k += 1
        else:
            self.process_id_ids_list_(in_list)

    def process_id_ids_list_(self, in_list):
        """

        :param in_list: [(id, [ids])]
        :return: None

        NB: according to current logic id 2 ids map is not kept
        but rather accumulated in prop_to_prop graph,
        thus all ids from in_list should already be processed
        by process_id_prop_list
        """

        # filter out pairs where id not in the set of ids
        in2 = filter(lambda x: x[0] in self.sets[id_type], in_list)
        # for every pair filter out the citing ids not in the set of ids
        in3 = map(lambda x: (x[0], list(filter(lambda y: y in self.sets[id_type],
                                               x[1]))), in2)
        # filter out pairs where the citing list is empty
        in4 = filter(lambda x: x[1], in3)

        if self.type_str[id_type]:
            in4 = list(map(lambda x: (self.str_to_int_maps[id_type][x[0]],
                                      list(map(lambda y: self.str_to_int_maps[id_type][y], x[1]))), in4))
        logging.info(' process_id_ids_list() : filter out pairs where id not in the set of ids')

        m_ida_idb, ida, idb = create_adj_matrix(in4)
        logging.info(' process_id_ids_list() : m_ida_idb shape {0}'.format(m_ida_idb.shape))

        nodes_type_a = list(map(lambda x: (id_type, x), ida))
        nodes_type_b = list(map(lambda x: (id_type, x), idb))

        m_prop_a_to_ida = project_graph_return_adj(self.g_prop_to_id, nodes_type_a, transpose=True)
        logging.info(' process_id_ids_list() : m_prop_a_to_ida shape {0}'.format(m_prop_a_to_ida.shape))
        m_idb_to_prop_b = project_graph_return_adj(self.g_prop_to_id, nodes_type_b)
        logging.info(' process_id_ids_list() : m_idb_to_prop_b shape {0}'.format(m_idb_to_prop_b.shape))
        prop_a = list(map(lambda x: ('{0}_a'.format(x[0]), x[1]), m_prop_a_to_ida.index))
        prop_b = list(map(lambda x: ('{0}_b'.format(x[0]), x[1]), m_idb_to_prop_b.columns))

        cc = dot(m_prop_a_to_ida, dot(m_ida_idb, m_idb_to_prop_b))
        ser = DataFrame(cc, index=prop_a, columns=prop_b).stack()
        df_prepared = ser[ser != 0.0].reset_index().rename(columns={0: 'weight'})
        prop_a_to_prop_b = from_pandas_adjacency(df_prepared, 'level_0', 'level_1', 'weight')
        logging.info(' process_id_ids_list() : cc {0}'.format(cc.shape))
        update_edges(self.g_prop_to_prop, prop_a_to_prop_b)
        logging.info(' process_id_prop_list() : prop_a_to_prop_b composition')
        logging.info(describe_graph(prop_a_to_prop_b))
        logging.info(describe_graph(self.g_prop_to_prop))
        logging.info(' process_id_ids_list() : {0} nodes, {1} edges in prop_to_prop'.format(len(prop_a_to_prop_b.nodes()),
                                                                    len(prop_a_to_prop_b.edges())))

    def update_sets_maps(self, new_items, key):
        outstanding = list(set(new_items) - self.sets[key])
        if outstanding:
            n = len(self.sets[key])
            self.sets[key].update(outstanding)
            logging.info(' update_sets_maps() : '
                         'sets[{0}] updated, currently holding {1} elements'.format(key, len(self.sets[key])))
            if self.type_str[key]:
                logging.info(' update_sets_maps() : key {0} is of str type'.format(key))
                outstanding_ints = list(range(n, n + len(outstanding)))
                logging.info(' update_sets_maps() : '
                             'i_to_str and str_to_i dicts will grow from {0} to {1}'.format(n, n + len(outstanding)))
                int_to_str_outstanding = dict(zip(outstanding_ints, outstanding))
                str_to_int_outstanding = dict(zip(outstanding, outstanding_ints))
                self.int_to_str_maps[key].update(int_to_str_outstanding)
                self.str_to_int_maps[key].update(str_to_int_outstanding)
                logging.info(' update_sets_maps() : '
                             'i_to_str len = {0}'.format(len(self.int_to_str_maps[key])))
                logging.info(' update_sets_maps() : '
                             'str_to_i len = {0}'.format(len(self.str_to_int_maps[key])))

    def g_props_to_df(self):
        logging.info(' in g_props_to_df() : describe prop to prop')
        logging.info(describe_graph(self.g_prop_to_prop))
        df = to_pandas_adjacency(self.g_prop_to_prop)
        index_ = sorted(list(filter(lambda x: x[0] == prop_type + '_b', df.index)), key=lambda x: x[1])
        columns_ = sorted(list(filter(lambda x: x[0] == prop_type + '_a', df.columns)), key=lambda x: x[1])

        df = df.loc[index_, columns_]
        df.rename(index=lambda x: x[1], columns=lambda x: x[1], inplace=True)

        sorted_props = sorted(list(set(df.index).union(set(df.columns))))
        df_tot = DataFrame(nan, columns=sorted_props, index=sorted_props)
        df_tot.update(df)
        logging.info(' j to j matrix shape {0}'.format(df_tot.shape))
        df = df_tot.fillna(0.)
        return df

    def retrieve_zij_counts_index(self):
        zij = self.g_props_to_df()
        common_index = list(set(self.prop_counts.index) | set(zij.index))

        df_tot = DataFrame(nan, columns=common_index, index=common_index)
        df_tot.update(zij)
        zij = df_tot.fillna(0.)

        vc_tot = Series(nan, index=common_index)
        vc_tot.update(self.prop_counts)
        vc_tot = vc_tot.fillna(0.)
        freqs = vc_tot/vc_tot.sum()
        return zij.values, freqs.values, common_index

    def info(self):
        logging.info(' {0} elements in ids set'.format(len(self.sets[id_type])))
        logging.info(' {0} elements in props set'.format(len(self.sets[prop_type])))
        logging.info(' {0} in int to str ids map'.format(len(self.int_to_str_maps[id_type])))
        logging.info(' {0} in int to str props map'.format(len(self.int_to_str_maps[prop_type])))
        logging.info(' {0} nodes, {1} edges in prop_to_id'.format(len(self.g_prop_to_id.nodes()),
                                                                 len(self.g_prop_to_id.edges())))
        logging.info(' {0} nodes, {1} edges in prop_to_prop'.format(len(self.g_prop_to_prop.nodes()),
                                                                   len(self.g_prop_to_prop.edges())))

        logging.info(' {0} unique props, {1} total prop counts in '
                     'prop_counts'.format(self.prop_counts.shape[0], self.prop_counts.sum()))


class AccumulatorOrgs(object):
    # maybe to lower before saving strings

    def __init__(self):
        self.keys = ['year', 'country', 'city',
                     'organizations_pref', 'organizations', 'full_address']
        self.sets = {k: set() for k in self.keys}
        self.int_to_str_maps = {k: dict() for k in self.keys}
        self.str_to_int_maps = {k: dict() for k in self.keys}
        self.paths_set = set()
        self.type_str = dict(zip(self.keys, [True] * len(self.keys)))
        self.type_str['year'] = False
        self.i2key = dict(zip(list(range(len(self.keys))), self.keys))

    def info(self):
        for k in self.keys:
            logging.info(' AccumulatorOrgs.info() : size of set {0} is {1}'.format(k, len(self.sets[k])))

    def update_set_map(self, new_items, key):
        outstanding = list(set(new_items) - self.sets[key])
        if outstanding:
            n = len(self.sets[key])
            self.sets[key].update(outstanding)
            if self.type_str[key]:
                outstanding_ints = list(range(n, n + len(outstanding)))
                int_to_str_outstanding = dict(zip(outstanding_ints, outstanding))
                str_to_int_outstanding = dict(zip(outstanding, outstanding_ints))
                self.int_to_str_maps[key].update(int_to_str_outstanding)
                self.str_to_int_maps[key].update(str_to_int_outstanding)

    def update_sets_maps(self, flat):
        for j in range(len(self.keys)):
            self.update_set_map(list(map(lambda x: x[j], flat)), self.i2key[j])

    def accumulate_paths(self, flat):
        for item in flat:
            int_item = [self.str_to_int_maps[k][x]
                        if self.type_str[k] else x for (k, x) in zip(self.keys, item)]
            self.paths_set.update({tuple(int_item)})

    def flatten_acc(self, acc):
        flat_list = []
        for item in acc:
            year = item['date']['year']
            for addr in item['addresses']:
                cnt = addr['country']
                city = addr['city']
                orgs_pref = tuple(sorted(addr['organizations_pref']))
                orgs = tuple(sorted(addr['organizations']))
                faddr = addr['full_address']
                flat_list.append((year, cnt, city, orgs_pref, orgs, faddr))
        return flat_list

    def process_acc(self, acc):
        flat = self.flatten_acc(acc)
        self.update_sets_maps(flat)
        self.accumulate_paths(flat)

    def dump(self, fpath):
        output = {'sets': self.sets,
                  'maps': {'i2s': self.int_to_str_maps, 's2i': self.str_to_int_maps},
                  'types': self.type_str,
                  'paths': array(list(self.paths_set))}

        with gzip.open(fpath, 'wb') as fp:
            pickle.dump(output, fp)


class AccumulatorCite(object):
    def __init__(self, fname=None, economical_mode=True):
        self.set_str_ids = set()
        self.int_to_str_map = dict()
        self.str_to_int_map = dict()
        # key : wid // int  ; value : (year, month, date) // (int, int, int)
        self.id_date = dict()
        # key : wid // int  ; value : [wid] //[int]
        self.id_cited_by = dict()
        self.economical_mode = economical_mode
        self.loaded = False
        self.fname = fname

    def info(self):
        logging.info(' AccumulatorCite.info() : obj {0}'.format(self.fname, len(self.str_to_int_map)))
        logging.info(' AccumulatorCite.info() : number of entries {0} or {1:.1f}M'.format(len(self.str_to_int_map),
                                                                                          len(self.str_to_int_map)/1e6))
        s = sum([len(v) for v in self.id_cited_by.values()])
        logging.info(' AccumulatorCite.info() : number of citations {0} or {1:.1f}M'.format(s, s/1e6))
        size_a = asizeof(self) / 1024 ** 3
        logging.info(' AccumulatorCite.info() : memsize {0:.2f} Gb'.format(size_a))

    def update_set_map(self, new_items):
        outstanding = [k for k in new_items if k not in self.str_to_int_map.keys()]
        if outstanding:
            n = len(self.str_to_int_map)
            outstanding_ints = list(range(n, n + len(outstanding)))
            str_to_int_outstanding = dict(zip(outstanding, outstanding_ints))
            self.str_to_int_map.update(str_to_int_outstanding)
            if not self.economical_mode:
                self.set_str_ids.update(outstanding)
                int_to_str_outstanding = dict(zip(outstanding_ints, outstanding))
                self.int_to_str_map.update(int_to_str_outstanding)
        gc.collect()

    def update_dates(self, items, item_dates, priority=True, check_int_ids=True):
        if check_int_ids:
            self.update_set_map(items)
        items_int = [self.str_to_int_map[s] for s in items]
        if priority:
            # update unconditionally
            # it is implied that item_dates contains triplets of year, month, day
            oustanding_data_dict = dict(zip(items_int, item_dates))
        else:
            # update only if id in not in
            outstanding = [k for k in items_int if k not in self.id_date.keys()]
            oustanding_data_dict = dict([(item, date) for item, date
                                         in zip(items_int, item_dates) if item in outstanding])
        self.id_date.update(oustanding_data_dict)
        gc.collect()

    def update_citations(self, cdata, check_int_ids=True, verbose=False):
        if check_int_ids:
            prim_ids = [item[0] for item in cdata]
            ref_id_year_ss = [item[1] for item in cdata]
            ref_ids = [ref_id for sublist in ref_id_year_ss for ref_id, y in sublist]
            self.update_set_map(prim_ids + ref_ids)

        for i_str, refs in cdata:
            if verbose:
                print(i_str, refs)
            id_int = self.str_to_int_map[i_str]
            refs_int = [self.str_to_int_map[j] for j, y in refs]

            for r in refs_int:
                if r in self.id_cited_by:
                    self.id_cited_by[r] += [id_int]
                else:
                    self.id_cited_by[r] = [id_int]

        gc.collect()

    def _update_citations(self, cdict):

        for k, v in cdict.items():
            if k in self.id_cited_by.keys():
                cdict[k] += self.id_cited_by[k]
        self.id_cited_by.update(cdict)
        gc.collect()

    def _update_dates(self, date_dict):
        for k in list(date_dict.keys()):
            if k in self.id_date.keys() and not (date_dict[k][1] and date_dict[k][2]):
                del date_dict[k]
        self.id_date.update(date_dict)
        gc.collect()

    def load(self, fpath=None, economical_mode=True, str_to_byte=True, strip_prefix='WOS:'):
        self.economical_mode = economical_mode
        if fpath:
            self.fname = fpath
        with gzip.open(self.fname, 'rb') as fp:
            pack = pickle.load(fp)

        self.id_cited_by = pack['id_cited_by']
        self.id_cited_by = {k: list(v) for k, v in self.id_cited_by.items()}
        pp = pack['maps']['s2i']
        if strip_prefix:
            lenp = len(strip_prefix)
            pp = {k[lenp:] if k[:lenp] == strip_prefix else k: v for k, v in pp.items()}

        first_key = next(iter(pp.keys()))
        if str_to_byte and not(is_bstr(pp[first_key])):
            pp = {k.encode('latin-1'): v for k, v in pp.items()}

        self.str_to_int_map = pp

        if not self.economical_mode:
            self.int_to_str_map = pack['maps']['i2s']
            self.set_str_ids = pack['set_wos_ids']
        self.id_date = pack['id_date']
        self.loaded = True
        gc.collect()

    def dump(self, fpath, economical_mode=True):
        self.economical_mode = economical_mode

        self.id_cited_by = {k: set(v) for k, v in self.id_cited_by.items()}

        output = {
                  'maps': {'s2i': self.str_to_int_map},
                  'id_cited_by': self.id_cited_by,
                  'id_date': self.id_date
                  }

        if not self.economical_mode:
            output['maps']['i2s'] = self.int_to_str_map
            output['set_wos_ids'] = self.set_str_ids

        with gzip.open(fpath, 'wb') as fp:
            pickle.dump(output, fp)

    def update_with_pub_data(self, id_data, date_data):
        year_data = [None if 'year' not in x.keys() else x['year'] for x in date_data]
        month_data = [None if 'month' not in x.keys() else x['month'] for x in date_data]
        day_data = [None if 'day' not in x.keys() else x['day'] for x in date_data]
        self.update_dates(id_data, zip(year_data, month_data, day_data), False)
        gc.collect()

    def update_with_cite_data(self, cite_data):

        id_data = [x for x, y in cite_data]
        self.update_set_map(id_data)
        del id_data

        # lists of refs (id_str, y)
        for wid, item in cite_data:
            wids = [x for x, y in item]
            self.update_set_map(wids)
            ref_dates = [(y, None, None) for x, y in item]
            self.update_dates(wids, ref_dates, False, False)
            del wids
            del ref_dates

        # flat list of refs
        self.update_citations(cite_data, False)
        gc.collect()

    def merge(self, b):
        """

        :param a: self
        :param b: AccumulatorCite obj that is merged onto a
        :return:
        """
        # if isinstance(b, AccumulatorCite):
        # a <= a, b
        a = self

        wids_new = [k for k in b.str_to_int_map.keys() if k not in a.str_to_int_map.keys()]
        a.update_set_map(wids_new)

        int_ids_b = list(b.str_to_int_map.values())
        if b.economical_mode:
            b.int_to_str_map = {k: v for v, k in b.str_to_int_map.items()}

        int_int_map_ba = {ib: a.str_to_int_map[b.int_to_str_map[ib]] for ib in int_ids_b}

        id_cited_by_conv = {int_int_map_ba[k]: [int_int_map_ba[x] for x in list(v)]
                            for k, v in b.id_cited_by.items()}

        a._update_citations(id_cited_by_conv)

        id_date_conv = {int_int_map_ba[k]: d for k, d in b.id_date.items()}

        a._update_dates(id_date_conv)

        return self

