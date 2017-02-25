from networkx import Graph
from graph_tools.reduce import reduce_bigraphs, update_edges


class Accumulator(object):

    def __init__(self, id_type_str=False, prop_type_str=False):

        self.type = ('id', 'j')
        # is_a_string_type flags
        self.type_str = {'ids': id_type_str, 'props': prop_type_str}
        # set of ids
        self.sets = {'ids': set(), 'props': set()}
        # x^str -> x^int (x int to x str dict) x ~ id, prop
        self.int_to_str_maps = {'ids': {}, 'props': {}}
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
        if self.type_str['ids']:
            self.update_int_str_map(new_ids, 'ids')
        if self.type_str['props']:
            self.update_int_str_map(new_props, 'props')

        # obtain int-int list
        new_ids_transformed = map(lambda x: x if not self.type_str['ids']
                                  else self.int_to_str_maps['ids'][x], new_ids)
        new_props_transformed = map(lambda x: x if not self.type_str['props']
                                    else self.int_to_str_maps['ids'][x], new_props)
        in_list_transformed = zip(new_ids_transformed, new_props_transformed)
        # update self.prop_to_id = Graph()
        for id_, prop_ in in_list_transformed:
            self.g_prop_to_id.add_edge((self.type[0], id_),
                                       (self.type[1], prop_), {'weight': 1.0})

    def process_id_ids_list(self, in_list):
        """

        :param in_list: [(id, [ids])]
        :return: None
        """
        ids_all = list(map(lambda x: x[0], in_list))
        for _, sublist in in_list:
            ids_all.extend(sublist)

        if self.type_str['ids']:
            self.update_int_str_map(ids_all, 'ids')

        ids = map(lambda x: x[0], in_list)

        ids_transformed = map(lambda x: x if not self.type_str['ids']
                              else self.int_to_str_maps['ids'][x], ids)

        refs_transformed = []
        for _, sublist in in_list:
            sub_transformed = map(lambda x: x if not self.type_str['ids']
                                  else self.int_to_str_maps['ids'][x], sublist)
            refs_transformed.extend(sub_transformed)

        # w -> u
        g_refs = Graph()
        # update self.prop_to_id = Graph()
        for id_, refs_ in zip(ids_transformed, refs_transformed):
            for item_ in refs_:
                g_refs.add_edge(('w', id_), ('u', item_), {'weight': 1.0})

        # iw, ju, wu
        # i -> w -> u; j -> u; i -> j
        # ju = gea.generate_bigraph(('j', 'u'), (journals_u, pubsU))
        # iw = gea.generate_bigraph(('i', 'w'), (journals_w, pubsW))
        # wu = gea.generate_bigraph(('w', 'u'), (pubsW, pubsU))
        # jw = gr.reduce_bigraphs(ju, wu)
        # ij = gr.reduce_bigraphs(iw, jw)

        jw = reduce_bigraphs(self.g_prop_to_id, g_refs)
        ij = reduce_bigraphs(self.g_prop_to_id, jw)
        update_edges(self.g_prop_to_prop, ij)

    def update_int_str_map(self, new_items, key):
        outstanding = list(set(new_items) - self.sets[key])
        if outstanding:
            n = len(self.sets[key])
            int_to_str_outstanding = dict(zip(range(n, n + len(outstanding)),
                                              list(outstanding)))
            self.sets[key].update(outstanding)
            self.int_to_str_maps[key].update(int_to_str_outstanding)

