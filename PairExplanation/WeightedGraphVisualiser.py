import heapq
import random
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import numpy as np

import utils


class WeightedGraphVisualiser:
    show_nodes: bool
    show_proportional_nodes: bool
    font_size: int
    percentage_of_nodes_to_show: float

    def __init__(self,
                 show_nodes: bool = False,
                 show_proportional_nodes: bool = True,
                 font_size: int = 15,
                 only_show_percentage: float = 0.20):
        self.show_nodes = show_nodes
        self.show_proportional_nodes = show_proportional_nodes
        self.font_size = font_size
        self.percentage_of_nodes_to_show = only_show_percentage

    def list_of_strings_to_label_dict(self, node_names: list[str]) -> dict[int, str]:
        # also note that the nodes are 1 indexed???
        return {(index): name for index, name in enumerate(node_names)}

    def weights_to_edge_thicknesses(self, original_weights_dict: dict[(int, int), float]) -> list[float]:
        original_items = list(original_weights_dict.items())
        original_nodes, original_weights = utils.unzip(original_items)
        new_weights = utils.remap_array(np.array(original_weights), new_min=1, new_max=5)
        return list(new_weights)

    def get_edge_thicknesses_of_graph(self, graph: nx.Graph):
        old_edge_weights = nx.get_edge_attributes(graph, 'weight')
        return self.weights_to_edge_thicknesses(old_edge_weights)

    def get_graph_and_positions(self, weight_matrix: np.ndarray) -> (nx.Graph, Any):
        plt.clf()
        plt.figure(figsize=(5, 5))
        important_connections = self.get_important_connections_from_weight_matrix(weight_matrix)

        graph = nx.Graph()
        n = weight_matrix.shape[0]
        #graph.add_nodes_from(range(1, n+1))  # use this to forcefully include the nodes
        for node_a, node_b, weight in important_connections:
            graph.add_edge(node_a, node_b, weight=weight)

        positions = nx.spring_layout(graph, iterations = 50)

        return graph, positions

    def draw_fancy_edges(self, graph, positions):
        # Draw edge labels to show weights
        edge_labels = nx.get_edge_attributes(graph, 'weight')

        # make the labels of the edges stand out against the edges by adding a background
        nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_color='blue',
                                     bbox=dict(facecolor='white', edgecolor='none', alpha=1.0))

    def draw_fancy_nodes(self, graph: nx.Graph, positions, node_name_dict):
        # draws the text in bold and with a near-white background
        labels = self.list_of_strings_to_label_dict(node_name_dict)

        # then we have to remove the labels for the nodes which are not included in the graph
        new_labels = {prev_key: prev_value
                      for prev_key,prev_value in labels.items()
                      if prev_key in positions}

        nx.draw_networkx_labels(graph, positions, labels=new_labels,
                                font_color='black', font_size=self.font_size,
                                font_weight='bold',
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))


    def get_importance_of_nodes_as_colours(self, weight_matrix: np.ndarray, graph: nx.Graph) -> list:
        importances = np.diag(weight_matrix)
        norm = mcolors.Normalize(vmin=min(importances), vmax=max(importances))
        white_to_green = mcolors.LinearSegmentedColormap.from_list("WhiteToGreen", ["white", "green"])
        node_colors = [white_to_green(norm(importances[n])) for n in graph.nodes]
        return node_colors

    def make_plot(self, weight_matrix: np.ndarray, node_names: list[str]):
        graph, positions = self.get_graph_and_positions(weight_matrix)

        edge_thicknesses = self.get_edge_thicknesses_of_graph(graph)
        node_colours = self.get_importance_of_nodes_as_colours(weight_matrix, graph)
        nx.draw(graph, positions,
                width=edge_thicknesses,
                node_color = node_colours if self.show_proportional_nodes else 'white',
                edge_color='grey',
                with_labels=False,  # because we're going to draw over them anyway
                node_size=1000)

       # self.draw_fancy_edges(graph, positions)
        self.draw_fancy_nodes(graph, positions, node_names)

        plt.title("Weighted Graph Visualization")
        return plt

    def get_important_connections_from_weight_matrix(self, weight_matrix: np.ndarray) -> list[(int, int, float)]:
        node_count = weight_matrix.shape[0]
        # for now we ignore reflection
        all_connections = [(a, b, weight_matrix[a, b])
                           for a in range(node_count)
                           for b in range(a + 1, node_count)]

        qty_nodes_kept = round(node_count * node_count * self.percentage_of_nodes_to_show)
        return heapq.nlargest(qty_nodes_kept, all_connections, key=utils.third)


def main():
    n = 5
    weight_matrix = np.zeros(shape=(n, n))

    for _ in range(12):
        x = random.randrange(n)
        y = random.randrange(n)
        weight = random.randrange(4, 12)
        weight_matrix[x, y] = weight
        weight_matrix[y, x] = weight

    graph_visualiser = WeightedGraphVisualiser()
    names = ["Lydia", "Hannah", "Zoe", "Louise", "Boden"]
    plot = graph_visualiser.make_plot(weight_matrix, names)
    plot.show()

