import itertools
import json
import os
import random
import re
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
    def from_json(cls, problem_file: str):
        with open(problem_file, "r") as file:
            data = json.load(file)

        return cls(amount_of_colours=data["amount_of_colours"],
                   amount_of_nodes=data["amount_of_nodes"],
                   connections=data["connections"])

    @classmethod
    def from_file(cls, problem_file: str):
        return cls.from_json(problem_file)

    def get_descriptors_of_ps(self, ps: PS) -> dict:
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

    def repr_property_new(self, property_name: str, property_value: float, rank: (float, float), ps: PS):
        # lower_rank, upper_rank = property_rank_range
        is_low = rank < 0.5
        rank_str = f"(rank = {int(rank * 100)}%)"  # "~ {int(property_rank_range[1]*100)}%)"

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
                result.values[(island_index * colour_count):((island_index + 1) * colour_count)] = colour_combo
                return result

            colour_combinations = itertools.permutations(list(range(colour_count)))
            return map(make_ps_for_colour_combination, colour_combinations)

        target_pss = {target for island_index in range(amount_of_islands)
                      for target in make_targets_for_island(island_index)}

        return cls(amount_of_colours=colour_count,
                   amount_of_nodes=amount_of_nodes,
                   connections=connections,
                   target_pss=target_pss)

    def get_targets(self) -> set[PS]:
        if self.target_pss is None:
            raise Exception("Requesting the target PSs from a GC instance with unknown targets")
        else:
            return self.target_pss

    def get_short_code(self) -> str:
        return "GC"

    @classmethod
    def load_from_cnf_file(cls, cnf_file_location: str, qty_colours: int = 3):
        qty_nodes = None
        qty_edges = None
        edge_pairs = []

        with open(cnf_file_location, "r") as file:

            for line in file.readlines():
                if len(line) == 0 or line[0] == "c":
                    continue

                if line[0] == "p":
                    print("Found the problem line")
                    p_char, problem_kind, var_str, clause_str = line.split()
                    qty_nodes = int(var_str)
                    qty_edges = int(clause_str)

                if line[0] == "e":
                    line_contents = line[2:]
                    # the values in the line are 1-indexed
                    a, b = [int(value_str) for value_str in line_contents.split()]
                    edge_pairs.append((a - 1, b - 1))

            return cls(amount_of_nodes=qty_nodes,
                       amount_of_colours=qty_colours,
                       connections=edge_pairs)

    def to_json(self, path: str):
        with utils.open_and_make_directories(path) as file:
            data = {"amount_of_colours": self.amount_of_colours,
                    "amount_of_nodes": self.amount_of_nodes,
                    "connections": self.connections}
            json.dump(data, file, indent=4)


def convert_problem_files_from_cnf():
    file_names = ["anna.col", "jean.col"]
    root_directory = r"/Users/gian/PycharmProjects/PS-descriptors/resources/problem_definitions/GC"

    for file_name in file_names:
        full_path = os.path.join(root_directory, file_name)
        problem = GraphColouring.load_from_cnf_file(full_path)
        new_file_name = full_path + ".json"
        problem.to_json(new_file_name)


# convert_problem_files_from_cnf()


class GraphColouringPrettifier:
    abbreviation_list: list[str]
    abbreviation_dict: dict[str, str]
    connections_by_chapter: dict[(str, str), str]

    def __init__(self,
                 abbreviation_list,
                 abbreviation_dict,
                 connections_by_chapter):
        self.abbreviation_list = abbreviation_list
        self.abbreviation_dict = abbreviation_dict
        self.connections_by_chapter = connections_by_chapter

    def get_abbreviation_from_node_number(self, node_number: int) -> str:
        return self.abbreviation_list[node_number]

    def get_name_from_abbreviation(self, abbr: str) -> str:
        return self.abbreviation_dict[abbr]

    @classmethod
    def from_json(cls, json_file_name: str):
        with open(json_file_name, "r") as file:
            data = json.load(file)
        return cls(abbreviation_list=data["abbreviation_list"],
                   abbreviation_dict=data["abbreviation_dict"],
                   connections_by_chapter=data["connections_by_chapter"])

    def store_as_json(self, json_file_name: str):
        data = {"abbreviation_list": self.abbreviation_list,
                "abbreviation_dict": self.abbreviation_dict,
                "connections_by_chapter": self.connections_by_chapter}

        with utils.open_and_make_directories(json_file_name) as file:
            json.dump(data, file, indent=4)

    @classmethod
    def from_dat_file(cls, dat_file_name: str):
        # comment_line_pattern = re.compile(r"^\*(.*)")
        name_line_pattern = re.compile(r"^([A-Z]{1,}) (.*)")
        # chapter_line_pattern = re.compile(r"(\d+\.)+:")

        # comment_lines = []
        name_list = []
        name_dict = dict()
        # chapter_lines = []

        with open(dat_file_name, "r") as file:
            for line in file:
                match = name_line_pattern.match(line)
                if match:
                    abbr = match.group(1)
                    full_name = match.group(2)
                    name_list.append(abbr)
                    name_dict[abbr] = full_name

        return cls(abbreviation_list=name_list,
                   abbreviation_dict=name_dict,
                   connections_by_chapter=None)

    def repr_ps(self, ps: PS) -> str:
        abbreviations = [self.abbreviation_list[index] for index in ps.get_fixed_variable_positions()]
        full_names = list(map(self.get_name_from_abbreviation, abbreviations))
        colours = ["Red", "Green", "Blue", "Yellow"]
        chosen_colours = [colours[ps.values[index]] for index in ps.get_fixed_variable_positions()]
        return " + ".join(f"[{colour}]\t{name}" for name, colour in zip(full_names, chosen_colours))


def convert_dat_files():
    #gc_folder = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\problem_definitions\GC"
    gc_folder = r"/Users/gian/PycharmProjects/PS-descriptors/resources/problem_definitions/GC/"
    file_names = ["anna.dat", "jean.dat"]

    for file_name in file_names:
        full_path = os.path.join(gc_folder, file_name)
        gcp = GraphColouringPrettifier.from_dat_file(full_path)

        converted_file_name = full_path + ".json"
        gcp.store_as_json(converted_file_name)


def test_gcp():
    problem_file_name = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\problem_definitions\GC\jean.json"
    dat_json_file_name = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\problem_definitions\GC\jean.dat.json"
    problem = GraphColouring.from_json(problem_file_name)
    gcp = GraphColouringPrettifier.from_json(dat_json_file_name)


    def random_small_ps():
        result = PS.empty(problem.search_space)
        for _ in range(random.randrange(3, 7)):
            result = result.with_fixed_value(random.randrange(problem.amount_of_nodes), random.randrange(3))
        return result

    pss = [random_small_ps() for _ in range(12)]
    for ps in pss:
        print(ps)
        print(gcp.repr_ps(ps))
        print("\n\n\n\n")


def interactive_checker():
    problem_file_name = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\problem_definitions\GC\jean.json"
    dat_json_file_name = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\problem_definitions\GC\jean.dat.json"
    problem = GraphColouring.from_json(problem_file_name)
    gcp = GraphColouringPrettifier.from_json(dat_json_file_name)

    def get_person() -> int:
        how = input("How would you like to find the person?")
        if how == "initials":
            initials = input("input the initials:")
            index = gcp.abbreviation_list.index(initials)
            print(f"The index is {index} for {initials}, the name is {gcp.abbreviation_dict[initials]}")
        elif how == "name":
            partial_name = input("input part of the name")
            winner_abbr = None
            for abbr, name in gcp.abbreviation_dict.items():
                if name.find(partial_name) >= 0:
                    print(f"Found for {name = }, {abbr = }")
                    if winner_abbr is None:
                        winner_abbr = abbr
                    else:
                        print("That pattern is present in multiple places... please retry")
                        return get_person()
            if winner_abbr is None:
                print("Could not find the pattern, please retry")
                return get_person()

            index = gcp.abbreviation_list.index(winner_abbr)
            print(f"The index is {index} for {winner_abbr} = {gcp.abbreviation_dict[winner_abbr]}")
        else:
            index = int(input("Insert the number directly "))
            return index

    while True:
        print("Getting person a")
        person_a = get_person()
        print("Getting person b")
        person_b = get_person()
        if ((person_a, person_b) in problem.connections) or ((person_b, person_a) in problem.connections):
            print("They are connected!")
        else:
            print("They are not connected")



# interactive_checker()