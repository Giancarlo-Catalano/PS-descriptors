import random

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PSMetric.Additivity import MutualInformation
from Core.PSMetric.MeanFitness import MeanFitness
from Core.PSMetric.Simplicity import Simplicity
from OwnLCS.OnlineLearner import OnlineLearner, OutputClass
from OwnLCS.RuleEvaluator import RuleEvaluator
from OwnLCS.RuleEvolver import RuleEvolver
from OwnLCS.RuleManager import RuleManager


def test_LCS(optimisation_problem: BenchmarkProblem,
             rule_population_size: int,
             solution_count: int,
             training_repeats: int):

    print("Genertating the PRef")
    pRef = optimisation_problem.get_reference_population(10000)

    print("Constructing the rule evaluator")
    rule_evaluator = RuleEvaluator.from_pRef(pRef)

    print("Constructing the rule evolver")
    rule_evolver = RuleEvolver(population_size=rule_population_size,
                               search_space=optimisation_problem.search_space,
                               rule_evaluator=rule_evaluator)

    print("Constructing the rule_manager")
    rule_manager = RuleManager(rule_evolver=rule_evolver,
                               rule_population=[],
                               evolve_new_individuals_interval=10,
                               subsumption_interval=20,
                               truncation_selection_interval=20)

    print("Constructing the online learner")
    #initial_solutions = [FullSolution.random(optimisation_problem.search_space) for _ in range(6)]
    learner = OnlineLearner(solutions_and_fitnesses=[], #[EvaluatedFS(s, optimisation_problem.fitness_function(s)) for s in initial_solutions],
                            rule_manager=rule_manager,
                            tournaments_per_new_solution = training_repeats)


    def train_with_tournaments():

        print("Training the learner using tournaments")
        for instance_number in range(solution_count):
            random_solution = FullSolution.random(optimisation_problem.search_space)
            fitness = optimisation_problem.fitness_function(random_solution)
            evaluated_solution = EvaluatedFS(random_solution, fitness)
            learner.pass_new_solution(evaluated_solution)

            if instance_number % 100 == 99:
                print(f"Progress:{round((instance_number / solution_count) * 100)}%")
                rule_manager.rule_population = rule_manager.apply_subsumption(rule_manager.rule_population)


    def train_using_percentages(batches: int = 1):
        print("Training the learner using percentages")
        pRef = optimisation_problem.get_reference_population(solution_count)

        # adjust the fitnesses
        values_present = sorted(list(set(pRef.fitness_array)))
        conversion_dict = {value: position/(len(values_present)-1)
                           for position, value in enumerate(values_present)}
        new_fitnesses = np.array([conversion_dict[value] for value in pRef.fitness_array])
        pRef.fitness_array = new_fitnesses

        adjusted_solutions = pRef.get_evaluated_FSs()


        def get_training_instance(e_fs: EvaluatedFS) -> (FullSolution, OutputClass):
            sent_class = random.random() < e_fs.fitness
            return (e_fs, sent_class)

        for batch_iteration in range(batches):
            print(f"Training on batch #{batch_iteration}")
            batch = list(map(get_training_instance, adjusted_solutions))
            learner.rule_manager.apply_training_batch(batch)

            # learner.rule_manager.apply_subsumption()


    train_using_percentages(training_repeats)
    # train_with_tournaments()


    print("Training has concluded")
    print("Now we test the learner")


    print("The rules at the end are")
    learner.rule_manager.rule_population.sort(key=lambda x:x.cached_linkage, reverse=True)
    for rule in learner.rule_manager.rule_population:
        print(rule)

    for _ in range(24):
        random_solution = FullSolution.random(optimisation_problem.search_space)
        fitness = optimisation_problem.fitness_function(random_solution)

        guessed_optimality = learner.guess_probability_of_optimality(random_solution)
        print(f"For the solution {optimisation_problem.repr_fs(random_solution)}")
        print(f"With fitness {fitness}")
        print(f"The learner guessed {guessed_optimality}")
