import networkx as nx
from typing import Dict, List
from collections import namedtuple
from collections import defaultdict
PairInteractionsData = namedtuple("PairInteractionsData", ["replies", 'quotes'])

class InteractionsGraph:
    WEIGHT_FIELD = "weight"
    def __init__(self):
        self.graph = nx.Graph()
        self.node_to_author = {}
        # self.size = len(self.node_to_author)

    def calc_weight(self,interactions: PairInteractionsData) -> float:
        weight = interactions[0]
        return  weight


    def add_interaction(self, author1: str, author2: str):
        if author1 not in self.graph:
            self.graph.add_node(author1)
            self.node_to_author[len(self.node_to_author)] = author1
        if author2:
          if author2 not in self.graph:
              self.graph.add_node(author2)
              self.node_to_author[len(self.node_to_author)] = author2

          if not self.graph.has_edge(author1, author2):
              # Add edge between nodes
              weight = 0
              self.graph.add_edge(author1, author2, **{self.WEIGHT_FIELD: weight})

          # Increment edge weight
          self.graph[author1][author2][self.WEIGHT_FIELD] += 1


    def set_interaction_weights(self):
        for u, v in self.graph.edges():
            weight_data = PairInteractionsData(
                replies=self.get_num_replies(u, v),
                quotes=0
            )
            weight = self.calc_weight(weight_data)
            self.graph[u][v][self.WEIGHT_FIELD] = weight

    def get_weighted_edges(self) -> List[Dict[str, int]]:
        edges = []

        for u, v, weight_dict in self.graph.edges(data=True):
            edges.append({
                "source": list(self.graph.nodes()).index(u),
                "target": list(self.graph.nodes()).index(v),
                "weight": weight_dict[self.WEIGHT_FIELD]
            })

        return edges

    def get_node_id_from_author(self, author: str) -> int:
        return list(self.graph.nodes()).index(author)

    def get_num_replies(self, author1: str, author2: str) -> int:
        if not self.graph.has_edge(author1, author2):
            return 0
        return self.graph[author1][author2][self.WEIGHT_FIELD] #this needs to be modified


    def get_author_from_node_id(self, node_id: int) -> str:
        return self.node_to_author[node_id]

    def get_core_interactions(self):
        core_nodes = set(nx.k_core(self.graph, k=2).nodes())
        core_edges = [(u, v, data) for u, v, data in self.graph.edges(data=True) if u in core_nodes and v in core_nodes]
        core_graph = InteractionsGraph()
        core_graph.graph.add_nodes_from(core_nodes)
        core_graph.graph.add_edges_from(core_edges)
        return core_graph

    def get_subgraph(self, nodes):
        """
        Returns a subgraph of the current graph containing only the specified nodes.

        Parameters:
            - nodes (list): A list of node IDs to include in the subgraph.

        Returns:
            - subgraph (InteractionsGraph): A new InteractionsGraph object representing the subgraph.
        """
        # Create a new InteractionsGraph object to represent the subgraph
        subgraph = InteractionsGraph()

        # Add each node in the specified list to the new graph
        for node in nodes:
            if node in self.graph.nodes:
                subgraph.graph.add_node(node)

        # Add each edge between nodes in the specified list to the new graph
        for edge in self.graph.edges:
            if edge[0] in nodes and edge[1] in nodes:
                weight = self.graph[edge[0]][edge[1]][InteractionsGraph.WEIGHT_FIELD]
                subgraph.graph.add_edge(edge[0], edge[1], weight=weight)

        return subgraph

    def parse_from_conv(self, conv):
        # for conv in convs:


        messages_by_author = defaultdict(list)
        for message in conv.messages:
            messages_by_author[message.author].append(message)

        for author1, messages1 in messages_by_author.items():
            for author2, messages2 in messages_by_author.items():
                if author1 != author2:
                    n_replies = sum(1 for m1 in messages1 for m2 in messages2 if m2.node_id == m1.parent_id)
                    weight = self.calc_weight(PairInteractionsData(replies=n_replies, quotes = 0))
                    for _ in range(weight):
                        self.add_interaction(author1, author2)
        # self.size = len(self.node_to_author)