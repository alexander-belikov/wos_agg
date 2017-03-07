from numpy import nan
from pandas import DataFrame, Series
from networkx import Graph, to_pandas_dataframe
from graph_tools.reduction import reduce_bigraphs, update_edges, describe_graph
import logging

id_type = 'id'
prop_type = 'prop'


class Accumulator(object):

    def __init__(self, id_type_str=False, prop_type_str=False):

        self.type = (id_type, prop_type)
        # is_a_string_type flags
        self.type_str = {id_type: id_type_str, prop_type: prop_type_str}
        # set of ids
        self.sets = {id_type: set(), prop_type: set()}

        # x^str -> x^int (x str to x int dict) x ~ id, prop
        self.str_to_int_maps = {id_type: {}, prop_type: {}}

        # x^int -> x^str (x int to x str dict) x ~ id, prop
        self.int_to_str_maps = {id_type: {}, prop_type: {}}

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
            self.update_prop_counts(new_props)

        logging.info('in process_id_prop_list() : g_prop_to_id composition')
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

    def process_id_ids_list(self, in_list):
        """

        :param in_list: [(id, [ids])]
        :return: None

        NB: according to current logic id 2 ids map is not kept
        but rather accumulated in prop_to_prop graph,
        thus all ids from in_list should already be processed
        by process_id_prop_list
        """

        in2 = filter(lambda x: x[0] in self.sets[id_type], in_list)

        in3 = map(lambda x: (x[0], list(filter(lambda y: y in self.sets[id_type],
                                               x[1]))), in2)
        in4 = filter(lambda x: x[1], in3)

        if self.type_str[id_type]:
            in4 = map(lambda x: (self.str_to_int_maps[id_type][x[0]],
                                 list(map(lambda y: self.str_to_int_maps[id_type][y], x[1]))), in4)

        # w -> u : w cites u's
        g_refs = Graph()
        # update self.prop_to_id = Graph()
        for id_, refs_ in in4:
            for item_ in refs_:
                g_refs.add_edge((id_type+'_A', id_), (id_type, item_), {'weight': 1.0})

        # print('in process_id_prop_list() : g_refs composition')
        # print(describe_graph(g_refs))

        # g_prop_to_id : self.type[0]//id, self.type[1]//prop
        # A -> B (A cites B)
        # j -> id; id_A -> id :  j_B -> id(A)
        prop_b_to_id_a = reduce_bigraphs(self.g_prop_to_id, g_refs,
                                         (prop_type + '_B', id_type))

        # print('in process_id_prop_list() : prop_b_to_id_a composition')
        # print(describe_graph(prop_b_to_id_a))

        # j -> id; j_B -> id :  j_A -> j_B
        prop_a_to_prop_b = reduce_bigraphs(self.g_prop_to_id, prop_b_to_id_a,
                                           (prop_type + '_A', prop_type + '_B'))

        update_edges(self.g_prop_to_prop, prop_a_to_prop_b)
        logging.info('in process_id_prop_list() : prop_a_to_prop_b composition')
        logging.info(describe_graph(prop_a_to_prop_b))

        logging.info('{0} nodes, {1} edges in prop_to_prop'.format(len(prop_a_to_prop_b.nodes()),
                                                                   len(prop_a_to_prop_b.edges())))

    def update_sets_maps(self, new_items, key):
        outstanding = list(set(new_items) - self.sets[key])
        if outstanding:
            self.sets[key].update(outstanding)
            if self.type_str[key]:
                n = len(self.sets[key])
                outstanding_ints = list(range(n, n + len(outstanding)))
                int_to_str_outstanding = dict(zip(outstanding_ints, outstanding))
                str_to_int_outstanding = dict(zip(outstanding, outstanding_ints))

                self.int_to_str_maps[key].update(int_to_str_outstanding)
                self.str_to_int_maps[key].update(str_to_int_outstanding)

    def g_props_to_df(self):
        df = to_pandas_dataframe(self.g_prop_to_prop).sort_index()
        df = df.reindex_axis(sorted(df.columns), axis=1)
        index_ = list(filter(lambda x: x[0] == prop_type + '_B', df.index))
        columns_ = list(filter(lambda x: x[0] == prop_type + '_A', df.columns))
        df = df.loc[list(index_), list(columns_)]
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
        logging.info('{0} elements in ids set'.format(len(self.sets[id_type])))
        logging.info('{0} elements in props set'.format(len(self.sets[prop_type])))
        logging.info('{0} in int to str ids map'.format(len(self.int_to_str_maps[id_type])))
        logging.info('{0} in int to str props map'.format(len(self.int_to_str_maps[prop_type])))
        logging.info('{0} nodes, {1} edges in prop_to_id'.format(len(self.g_prop_to_id.nodes()),
                                                                 len(self.g_prop_to_id.edges())))
        logging.info('{0} nodes, {1} edges in prop_to_prop'.format(len(self.g_prop_to_prop.nodes()),
                                                                   len(self.g_prop_to_prop.edges())))

        logging.info('{0} unique props, {1} total prop counts in '
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

    def process_acc(self, acc):
        list_of_lists = map(lambda x: list(map(lambda y: (y['country'], y['city'],
                                                          tuple(sorted(y['organizations_pref'])),
                                                          tuple(sorted(y['organizations'])),
                                                          y['full_address']), x['addresses'])), acc)
        flat_list = [x for sublist in list_of_lists for x in sublist]
        flat_list2 = [list(zip(keys, x)) for x in flat_list]
        return flat_list2
