from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from LCS.DifferenceExplainer.DifferenceExplainer import DifferenceExplainer


def test_difference_explainer():
    # problem = RoyalRoad(4, 4)
    problem = GraphColouring.random(amount_of_colours=3, amount_of_nodes=6, chance_of_connection=0.3)
    problem.view()

    diff_explainer = DifferenceExplainer.from_folder(problem = problem,
                                                     folder = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\gc_explain",
                                                     verbose=True)

    diff_explainer.explanation_loop(amount_of_fs_to_propose=6, suppress_errors=False)


test_difference_explainer()
