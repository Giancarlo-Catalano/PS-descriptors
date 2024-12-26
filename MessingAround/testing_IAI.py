
from interpretableai import iai
#import interpretableai

import utils
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PRef import PRef
from Explanation.PRefManager import PRefManager
from VarianceDecisionTree.IAIDecisionTree import IAIDecisionTree


def test_IAI():
    problem = RoyalRoad(5)
    pRef = PRefManager.generate_pRef(problem, 10000, which_algorithm="GA")
    pRef = PRef.unique(pRef)

    train_pRef, test_pRef = pRef.train_test_split(test_size=0.2, random_state=42)

    train_X = train_pRef.full_solution_matrix
    train_y = train_pRef.fitness_array

    test_X = train_pRef.full_solution_matrix
    test_y = train_pRef.fitness_array

    grid = iai.GridSearch(
        iai.OptimalTreeRegressor(
            random_seed=123,
        ),
        max_depth=range(1, 6),
    )
    grid.fit(train_X, train_y)
    grid.get_learner()


    predicted_y = grid.predict(test_X)

    print("I'm producing a simple scatterplot!")
    utils.simple_scatterplot(x_label="actual",
                             y_label="predicted",
                             xs=test_y,
                             ys=predicted_y)

    grid.show_in_browser()


# test_IAI()


def test_wrapper():
    problem = EfficientBTProblem.from_default_files()
    pRef = PRefManager.generate_pRef(problem, 1000, which_algorithm="GA")
    pRef = PRef.unique(pRef)

    train_pRef, test_pRef = pRef.train_test_split(test_size=0.2, random_state=42)

    regressor_wrapper = IAIDecisionTree(maximum_depth=4)
    regressor_wrapper.train_from_pRef(train_pRef)
    mse = regressor_wrapper.get_mse_on_test_data(test_pRef)
    print(f"The mse is {mse}")


test_wrapper()
#interpretableai.get_machine_id()


# def install_julia():
#     #interpretableai.install_julia()
#     interpretableai.install_system_image(accept_license=True)

# install_julia()

#def attempt_julia():
    # print("Using julia!")
    # machine_id = iai.get_machine_id()
    # print(f"The machine id is {machine_id}")


#attempt_julia()