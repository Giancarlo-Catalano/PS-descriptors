from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
import numpy as np

from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace


class QAP(BenchmarkProblem):
    distance_matrix: np.ndarray
    transport_matrix: np.ndarray

    qty_nodes: int

    def __init__(self,
                 distance_matrix: np.ndarray,
                 transport_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
        self.transport_matrix = transport_matrix

        assert (len(self.distance_matrix.shape) == 2)
        assert (self.distance_matrix.shape[0] == self.distance_matrix.shape[1])
        assert (self.distance_matrix.shape == self.transport_matrix.shape)

        self.qty_nodes = self.distance_matrix.shape[0]

        super().__init__(SearchSpace.from_permuation_of(self.qty_nodes))

    def fitness_function(self, fs: FullSolution) -> float:
        node_order = self.get_node_order_from_fs(fs)

        # this is just a reshuffling of the weight matrix so that it now represents the weights between locations
        weights_between_positions = self.transport_matrix[node_order, :][:, node_order]
        return float(np.sum(self.distance_matrix * weights_between_positions))

    def get_node_order_from_fs(self, fs: FullSolution) -> list[int]:
        # uses stack encoding, as used in a normal tsp
        left_to_pick = list(range(self.qty_nodes))
        result = []
        for item in fs.values:
            result.append(left_to_pick.pop(item))

        # include the forced choice
        result.extend(left_to_pick)
        return result





def test_QAP():
    distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    weights = np.array([[0, 10, 1], [10, 0, 10], [1, 10, 0]])
    problem = QAP(distances, weights)

    trials = 6
    for _ in range(trials):
        solution = FullSolution.random(problem.search_space)
        fitness = problem.fitness_function(solution)

        node_order = problem.get_node_order_from_fs(solution)
        print(f"The fitness for {solution} is {fitness}, (it meant {node_order}")


test_QAP()
