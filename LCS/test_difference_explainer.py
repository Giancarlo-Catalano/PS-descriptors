from BenchmarkProblems.RoyalRoad import RoyalRoad
from LCS.DifferenceExplainer.DifferenceExplainer import DifferenceExplainer


def test_difference_explainer():
    problem = RoyalRoad(4, 4)

    diff_explainer = DifferenceExplainer.from_folder(problem = problem,
                                                     folder = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\rr_explain",
                                                     verbose=True)

    diff_explainer.explanation_loop(amount_of_fs_to_propose=6)


test_difference_explainer()
