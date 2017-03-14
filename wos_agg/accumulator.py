from numpy import nan
from pandas import DataFrame, Series
from networkx import Graph, to_pandas_dataframe, write_gpickle, from_pandas_dataframe
from graph_tools.reduction import update_edges, describe_graph, project_graph_return_adj
from graph_tools.adj_aux import create_adj_matrix
import logging
from numpy import dot, arange
from gc import collect

id_type = 'id'
prop_type = 'prop'


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
        logging.info(' ::: {0}'.format(sum(ser.isnull())))
        df_prepared = ser[ser != 0.0].reset_index().rename(columns={0: 'weight'})
        prop_a_to_prop_b = from_pandas_dataframe(df_prepared, 'level_0', 'level_1', 'weight')
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
        df = to_pandas_dataframe(self.g_prop_to_prop)
        index_ = sorted(list(filter(lambda x: x[0] == prop_type + '_b', df.index)), key=lambda x: x[1])
        columns_ = sorted(list(filter(lambda x: x[0] == prop_type + '_a', df.columns)), key=lambda x: x[1])
        logging.info('props to props shape {0}'.format(df.shape))

        df = df.loc[index_, columns_]
        df.rename(index=lambda x: x[1], columns=lambda x: x[1], inplace=True)

        sorted_props = sorted(list(set(df.index).union(set(df.columns))))
        df_tot = DataFrame(nan, columns=sorted_props, index=sorted_props)
        df_tot.update(df)
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


keys = ['country', 'city', 'organizations_pref', 'organizations', 'full_address']


class AccumulatorOrgs(object):

    def __init__(self):
        self.set_orgs = set()
        self.dict_orgs = dict()
        self.g = Graph()

    def update(self, inlist):
        g = self.g
        for x in inlist:
            co, ci, orgp, org, add = x
            if co not in g.nodes():
                g.add_node(co)
                if ci not in g.neighbors(co):
                    g.add_edge(co, ci)
                    if orgp not in g.neighbors(ci):
                        g.add_edge(ci, orgp)
                        if org not in g.neighbors(orgp):
                            g.add_edge(orgp, org)
                            if add not in g.neighbors(org):
                                g.add_edge(org, add)

    def info(self):
        self.loggin('AccumulatorOrgs.info() : {0}'.format(describe_graph(self.g)))

    def process_acc(self, acc):
        filtered_acc = filter(lambda x: 'addresses' in x.keys(), acc)
        list_of_lists = map(lambda x: list(map(lambda y: (y['country'], y['city'],
                                                          '|'.join(sorted(y['organizations_pref'])),
                                                          '|'.join(sorted(y['organizations'])),
                                                          y['full_address']), x['addresses'])), filtered_acc)
        flat_list = [x for sublist in list_of_lists for x in sublist]
        flat_list2 = [list(zip(keys, x)) for x in flat_list]
        return flat_list2

    def dump(self, fpath):
        write_gpickle(self.g, fpath)
