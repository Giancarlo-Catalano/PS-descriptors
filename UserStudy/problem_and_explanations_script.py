import os

import utils
from BenchmarkProblems.SimplifiedBTProblem.SimplifiedBTProblem import SimplifiedBTProblem
from Core.PRef import PRef
from Core.PS import PS
from Explanation.MinedPSManager import MinedPSManager
from Explanation.PRefManager import PRefManager
from Explanation.PSPropertyManager import PSPropertyManager
from VarianceDecisionTree.PSRegressionTree import PSSearchSettings, PSRegressionTree

instance_a_path = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\UserStudy\Instances\A"
instance_b_path = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\UserStudy\Instances\B"


def get_pRef_path(instance_path: str) -> str:
    return os.path.join(instance_path, "pRef.npz")


def get_problem_path(instance_path: str) -> str:
    return os.path.join(instance_path, "problem.json")


def get_decision_tree_path(instance_path: str) -> str:
    return os.path.join(instance_path, "decision_tree.json")


def get_ps_properties_path(instance_path: str) -> str:
    return os.path.join(instance_path, "ps_properties.csv")


def get_ps_property_manager(instance_path: str,
                            problem: SimplifiedBTProblem) -> PSPropertyManager:
    ps_property_path = get_ps_properties_path(instance_path)
    ps_property_manager = PSPropertyManager(problem=problem,
                                            property_table_file=ps_property_path,
                                            verbose=True,
                                            threshold=0.25)

    return ps_property_manager


def generate_control_data_for_ps_properties(ps_property_manager: PSPropertyManager,
                                            pss_to_analyse: list[PS]):
    mined_ps_manager = MinedPSManager(problem=ps_property_manager.problem,
                                      verbose=False)
    mined_ps_manager.cached_pss = pss_to_analyse
    mined_ps_manager.cached_control_pss = mined_ps_manager.generate_control_pss(samples_for_each_category=3000)
    ps_property_manager.generate_property_table_file(pss=pss_to_analyse,
                                                     control_pss=mined_ps_manager.cached_control_pss)


def prepare_data_for_instance(instance_path: str,
                                generate_pRef=False,
                                generate_decision_tree=False,
                                generate_properties_table=False, ):
    problem_path = get_problem_path(instance_path)
    problem = SimplifiedBTProblem.from_json(problem_path)

    pRef_path = get_pRef_path(instance_path)
    if generate_pRef:
        with utils.announce(f"generating the pRef, storing it in {pRef_path}"):
            pRef = PRefManager.generate_pRef(problem=problem,
                                             sample_size=10000,
                                             which_algorithm="GA")
            pRef.save(pRef_path)
    else:
        with utils.announce(f"Loading the pRef from {pRef_path}"):
            pRef = PRef.load(pRef_path)

    search_settings = PSSearchSettings(ps_search_budget=10,
                                       ps_search_population=10,
                                       metrics="simplicity variance ground_truth_atomicity",
                                       avoid_ancestors=True,
                                       original_problem=problem,
                                       culling_method="biggest",
                                       verbose=True)

    decision_tree_path = get_decision_tree_path(instance_path)
    if generate_decision_tree:
        decision_tree = PSRegressionTree(maximum_depth=2)
        decision_tree.search_settings = search_settings

        with utils.announce(f"training the decision tree, storing it at {decision_tree_path}"):
            decision_tree.train_from_pRef(pRef, random_state=42)
            decision_tree.to_file(decision_tree_path)
    else:
        with utils.announce(f"Loading the decision tree from {decision_tree_path}"):
            decision_tree = PSRegressionTree.from_file(decision_tree_path)

    ps_property_manager = get_ps_property_manager(instance_path, problem)

    if generate_properties_table:
        with utils.announce(
                f"Generating the control data for the instance, and storing it at {ps_property_manager.property_table_file}, and updating the decision tree"):
            generate_control_data_for_ps_properties(ps_property_manager=ps_property_manager,
                                                    pss_to_analyse=decision_tree.all_pss_as_list())

            decision_tree.add_properties_to_pss(ps_property_manager)
            decision_tree.to_file(decision_tree_path)


def prepare_data_for_instance_a(generate_pRef=True,
                                generate_properties_table=True,
                                generate_decision_tree=True):
    prepare_data_for_instance(instance_path=instance_a_path,
                              generate_pRef=generate_pRef,
                              generate_properties_table=generate_properties_table,
                              generate_decision_tree=generate_decision_tree)


def prepare_data_for_instance_b(generate_pRef=True,
                                generate_properties_table=True,
                                generate_decision_tree=True):
    problem_a = SimplifiedBTProblem.from_json(get_problem_path(instance_a_path))
    problem_b = SimplifiedBTProblem.from_json(get_problem_path(instance_b_path))

    pRef_b_path = get_pRef_path(instance_b_path)
    if generate_pRef:
        with utils.announce(f"Generating the pRef and storing at {pRef_b_path}"):
            pRef_a = PRef.load(get_pRef_path(instance_a_path))
            pRef_b = problem_b.get_permutated_pRef(pRef_a)
            pRef_b.save(pRef_b_path)
    else:
        with utils.announce(f"Loading the PRef from {pRef_b_path}"):
            pRef_b = PRef.load(pRef_b_path)

    decision_tree_b_path = get_decision_tree_path(instance_b_path)
    if generate_decision_tree:
        with utils.announce(f"Generating the decision tree and storing at {decision_tree_b_path}"):
            decision_tree_a = PSRegressionTree.from_file(get_decision_tree_path(instance_a_path))
            decision_tree_b = decision_tree_a.with_permutation(problem_b.original_indexes)
            decision_tree_b.to_file(decision_tree_b_path)
    else:
        with utils.announce(f"Loading the pRef from {decision_tree_b_path}"):
            decision_tree_b = PSRegressionTree.from_file(decision_tree_b_path)

    ps_property_manager_b = get_ps_property_manager(instance_b_path, problem_b)
    if generate_properties_table:
        with utils.announce(
                f"Generating the control data for the pss, stored at {ps_property_manager_b.property_table_file}"):
            generate_control_data_for_ps_properties(ps_property_manager_b,
                                                    pss_to_analyse=decision_tree_b.all_pss_as_list())
            decision_tree_b.add_properties_to_pss(ps_property_manager_b)
            decision_tree_b.to_file(decision_tree_b_path)
    else:
        pass  # the property manager will fetch the table by itself


def show_data_for_instance(instance_path: str):
    problem = SimplifiedBTProblem.from_json(get_problem_path(instance_path))
    pRef = PRef.load(get_pRef_path(instance_path))
    best_solution = pRef.get_best_solution()
    decision_tree = PSRegressionTree.from_file(get_decision_tree_path(instance_path))
    decision_tree.problem = problem  # for pretty printing

    print(f"The best solution has fitness {best_solution.fitness}, "
          f"and it is \n{problem.repr_ps(PS.from_FS(best_solution))}")

    print(f"The decision tree is")
    decision_tree.print_ASCII()


def prepare_a_b_data():
    # prepare_data_for_instance_a(generate_pRef=False,
    #                             generate_decision_tree=True,
    #                             generate_properties_table=True)
    #
    # prepare_data_for_instance_b(generate_pRef=True,
    #                             generate_decision_tree=True,
    #                             generate_properties_table=True)

    print("FOR INSTANCE A")
    show_data_for_instance(instance_a_path)

    print("FOR INSTANCE B")
    show_data_for_instance(instance_b_path)




instance_c_path = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\UserStudy\Instances\Constructed"
def prepare_constructed_problem_data():
    prepare_data_for_instance(instance_path = instance_c_path,
                              generate_pRef=True,
                                generate_decision_tree=True,
                                generate_properties_table=True)


    show_data_for_instance(instance_c_path)

prepare_constructed_problem_data()




