import itertools
import json
import random
from typing import TypeAlias, Iterable, Optional

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace
import networkx as nx
import matplotlib.pyplot as plt

Node: TypeAlias = int
Connection: TypeAlias = (Node, Node)


# Function to visualize an undirected graph given a list of edges (node pairs)
def visualize_undirected_graph(edges):
    # Create a NetworkX graph
    graph = nx.Graph()  # Create an undirected graph

    # Add edges to the graph
    graph.add_edges_from(edges)  # Add all edges at once

    # Try using the planar layout to minimize edge overlaps
    try:
        pos = nx.planar_layout(graph)  # Use the planar layout for a non-overlapping arrangement
    except nx.NetworkXException:
        # If the graph is not planar, fall back to the spring layout
        pos = nx.spring_layout(graph)  # Use the spring layout as an alternative

    # Draw the graph with a specific layout
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='gray', font_size=14)

    # Display the plot
    plt.title("Undirected Graph (Planar Layout)")
    plt.show()

    return graph  # Return the NetworkX graph object for further operations



class GraphColouring(BenchmarkProblem):
    amount_of_colours: int
    amount_of_nodes: int

    connections: list[Connection]

    target_pss: Optional[set[PS]]

    def __init__(self,
                 amount_of_colours: int,
                 amount_of_nodes: int,
                 connections: Iterable,
                 target_pss: Optional[set[PS]] = None):
        self.amount_of_colours = amount_of_colours
        self.amount_of_nodes = amount_of_nodes
        self.connections = [(a, b) for (a, b) in connections]

        self.target_pss = target_pss

        search_space = SearchSpace([amount_of_colours for _ in range(self.amount_of_nodes)])
        super().__init__(search_space)

    def __repr__(self):
        return f"GraphColouring(#colours = {self.amount_of_colours}, #nodes = {self.amount_of_nodes}), connections are {self.connections}"

    def long_repr(self) -> str:
        return self.__repr__() + "  " + ", ".join(f"{connection}" for connection in self.connections)

    @classmethod
    def random(cls, amount_of_nodes: int,
               amount_of_colours: int,
               chance_of_connection: float):
        connections = []
        for node_a, node_b in itertools.combinations(range(amount_of_nodes), 2):
            if random.random() < chance_of_connection:
                connections.append((node_a, node_b))

        return cls(amount_of_colours=amount_of_colours,
                   amount_of_nodes=amount_of_nodes,
                   connections=connections)

    def fitness_function(self, fs: FullSolution) -> float:
        return float(sum([1 for (node_a, node_b) in self.connections
                          if fs.values[node_a] != fs.values[node_b]]))

    def repr_ps(self, ps: PS) -> str:
        colours = ["red", "green", "blue", "yellow", "purple", "orange", "black", "white", "pink", "brown", "gray",
                   "cyan"]

        def repr_node_and_colour(node_index, colour_index: int):
            return f"#{node_index} = {colours[colour_index]}"

        return "\n".join([repr_node_and_colour(node, colour)
                          for node, colour in enumerate(ps.values)
                          if colour != STAR])


    def view(self):
        visualize_undirected_graph(self.connections)

    def save(self, filename: str):
        """ simply stores the connections as a json"""
        data = {"amount_of_nodes": self.amount_of_nodes,
                "amount_of_colours": self.amount_of_colours,
                "connections": self.connections}
        utils.make_folder_if_not_present(filename)
        with open(filename, "w+") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def from_file(cls, problem_file: str):
        with open(problem_file, "r") as file:
            data = json.load(file)

        return cls(amount_of_colours=data["amount_of_colours"],
                   amount_of_nodes=data["amount_of_nodes"],
                   connections = data["connections"])


    def ps_to_properties(self, ps: PS) -> dict:
        def is_internal_edge(pair):
            a, b = pair
            return ps[a] != STAR and ps[b] != STAR

        def is_extenal_edge(pair):
            a, b = pair
            return (ps[a] != STAR and ps[b] == STAR) or (ps[b] != STAR and ps[a] == STAR)

        internal_edge_count = len([pair for pair in self.connections if is_internal_edge(pair)])

        external_edge_count = len([pair for pair in self.connections if is_extenal_edge(pair)])

        return {"internal_edge_count": internal_edge_count,
                "external_edge_count": external_edge_count}


    def repr_property(self, property_name:str, property_value:float, rank:(float, float), ps: PS):
        #lower_rank, upper_rank = property_rank_range
        is_low = rank < 0.5
        rank_str = f"(rank = {int(rank * 100)}%)" # "~ {int(property_rank_range[1]*100)}%)"

        if property_name == "internal_edge_count":
            return f"The PS is {'NOT ' if is_low else ''}densely connected {rank_str}"
        elif property_name == "external_edge_count":
            return f"The PS is {'' if is_low else 'NOT '}isolated {rank_str}"
        else:
            raise ValueError(f"Did not recognise the property {property_name} in GC")


    @classmethod
    def make_insular_instance(cls, amount_of_islands: int):
        colour_count = 3
        amount_of_nodes = colour_count * amount_of_islands
        def make_connections_for_island(island_index: int) -> list[Connection]:
            a = island_index * 3 + 0
            b = island_index * 3 + 1
            c = island_index * 3 + 2
            return [(a, b), (b, c), (c, a)]


        connections = [connection
                       for island_index in range(amount_of_islands)
                       for connection in make_connections_for_island(island_index)]

        search_space = SearchSpace([colour_count for node in range(amount_of_nodes)])
        def make_targets_for_island(island_index: int) -> Iterable[PS]:
            colour_combinations = itertools.permutations(list(range(colour_count)))
            def make_ps_for_colour_combination(colour_combo) -> PS:
                result = PS.empty(search_space)
                result.values[(island_index*colour_count):((island_index+1)*colour_count)] = colour_combo
                return result
            colour_combinations = itertools.permutations(list(range(colour_count)))
            return map(make_ps_for_colour_combination, colour_combinations)

        target_pss = {target for island_index in range(amount_of_islands)
                      for target in make_targets_for_island(island_index)}

        return cls(amount_of_colours=colour_count,
                   amount_of_nodes=amount_of_nodes,
                   connections = connections,
                   target_pss=target_pss)



    def get_targets(self) -> set[PS]:
        if self.target_pss is None:
            raise Exception("Requesting the target PSs from a GC instance with unknown targets")
        else:
            return self.target_pss
