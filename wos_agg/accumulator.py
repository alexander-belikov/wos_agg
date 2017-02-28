from networkx import Graph
from graph_tools.reduce import reduce_bigraphs, update_edges

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

        # prop^i -> id^i (prop int to id int)
        self.g_prop_to_id = Graph()
        # prop^i(citing) -> prop^i(cited) (prop int to prop int)
        self.g_prop_to_prop = Graph()

    def process_id_prop_list(self, in_list):
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

    def process_id_ids_list(self, in_list):
        """

        :param in_list: [(id, [ids])]
        :return: None

        NB: according to current logic id 2 ids map is not kept
        but rather accumulated in prop_to_prop graph,
        thus all ids from in_list should already be processed
        by process_id_prop_list
        """
        # ids_all = list(map(lambda x: x[0], in_list))
        # for _, sublist in in_list:
        #     ids_all.extend(sublist)
        #
        # # ids incoming type is str : update the maps
        # # at the same time, we should already have them before
        # if self.type_str['ids']:
        #     self.update_maps(ids_all, 'ids')

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


        # g_prop_to_id : self.type[0]//id, self.type[1]//prop

        prop_b_to_id_a = reduce_bigraphs(self.g_prop_to_id, g_refs,
                                         (prop_type + '_B', id_type))
        prop_a_to_prop_b = reduce_bigraphs(self.g_prop_to_id, prop_b_to_id_a,
                                           (prop_type + '_A', prop_type + '_B'))

        print('{0} nodes, {1} edges in prop_to_prop'.format(len(prop_a_to_prop_b.nodes()),
                                                            len(prop_a_to_prop_b.edges())))

        update_edges(self.g_prop_to_prop, prop_a_to_prop_b)

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

    def info(self):
        print('{0} elements in ids set'.format(len(self.sets[id_type])))
        print('{0} elements in props set'.format(len(self.sets[prop_type])))
        print('{0} in int to str ids map'.format(len(self.int_to_str_maps[id_type])))
        print('{0} in int to str props map'.format(len(self.int_to_str_maps[prop_type])))
        print('{0} nodes, {1} edges in prop_to_id'.format(len(self.g_prop_to_id.nodes()),
                                                          len(self.g_prop_to_id.edges())))
        print('{0} nodes, {1} edges in prop_to_prop'.format(len(self.g_prop_to_prop.nodes()),
                                                            len(self.g_prop_to_prop.edges())))
