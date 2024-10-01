from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from LCS.DifferenceExplainer.DifferenceExplainer import DifferenceExplainer


def test_difference_explainer():
    problem = RoyalRoad(6, 4)
    #problem = GraphColouring.random(amount_of_colours=3, amount_of_nodes=6, chance_of_connection=0.3)
    # problem.view()
    #problem = EfficientBTProblem.from_default_files()

    folder = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations" + "\\" + problem.get_short_code()+"explain"
    # folder = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\heavy_BT_explain"

    diff_explainer = DifferenceExplainer.from_folder(problem = problem,
                                                     folder = folder,
                                                     verbose=True)

    diff_explainer.explanation_loop(amount_of_fs_to_propose=6, suppress_errors=False)


test_difference_explainer()
