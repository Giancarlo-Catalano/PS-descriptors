# Explainer? I barely know her!
import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PS import PS, contains
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PSMetric.Linkage.OutdatedLinkage import UnivariateLocalPerturbation, BivariateLocalPerturbation
from Core.PSMetric.FitnessQuality.MeanFitness import MeanFitness
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import SignificantlyHighAverage
from utils import indent


class Explainer:
    benchmark_problem: BenchmarkProblem  # used mainly for repr_pr
    ps_catalog: list[EvaluatedPS]
    pRef: PRef

    mean_fitness_metric: MeanFitness
    statistically_high_fitness_metric: SignificantlyHighAverage
    local_importance_metric: UnivariateLocalPerturbation
    local_linkage_metric: BivariateLocalPerturbation

    overall_average: float

    def __init__(self,
                 benchmark_problem: BenchmarkProblem,
                 ps_catalog: list[EvaluatedPS],
                 pRef: PRef):
        self.benchmark_problem = benchmark_problem
        self.ps_catalog = ps_catalog
        self.pRef = pRef
        self.overall_average = np.average(self.pRef.fitness_array)

        self.mean_fitness_metric = MeanFitness()
        self.statistically_high_fitness_metric = SignificantlyHighAverage()
        self.local_importance_metric = UnivariateLocalPerturbation()
        self.local_linkage_metric = BivariateLocalPerturbation()

        for metric in [self.mean_fitness_metric, self.statistically_high_fitness_metric, self.local_importance_metric,
                       self.local_linkage_metric]:
            metric.set_pRef(self.pRef)

    def t_test_for_mean_with_ps(self, ps: PS) -> (float, float):
        return self.statistically_high_fitness_metric.get_p_value_and_sample_mean(ps)

    def get_small_description_of_ps(self, ps: PS) -> str:
        p_value, _ = self.t_test_for_mean_with_ps(ps)
        observations, not_observations = self.pRef.fitnesses_of_observations_and_complement(ps)
        avg_when_present = np.average(observations)
        avg_when_absent = np.average(not_observations)
        return f"{self.benchmark_problem.repr_ps(ps)}, avg when present = {avg_when_present:.2f}, avg when absent = {avg_when_absent:.2f}, p-value = {p_value:e}"

    def local_explanation_of_full_solution(self, full_solution: FullSolution):
        contained_pss = [ps for ps in self.ps_catalog
                         if contains(full_solution, ps)
                         if not ps.is_empty()]
        #contained_pss = Explainer.only_non_obscured_pss(contained_pss)
        contained_pss.sort(reverse=True, key = lambda x: x.metric_scores[-1])  # sort by atomicity

        fs_as_ps = PS.from_FS(full_solution)
        print(f"The solution \n {indent(self.benchmark_problem.repr_ps(fs_as_ps))}\ncontains the following PSs:")
        for ps in contained_pss[:12]:
            print(indent(self.get_small_description_of_ps(ps)))
            print()

        # local_importances = self.local_importance_metric.get_local_importance_array(fs_as_ps)
        # local_linkages = self.local_linkage_metric.get_local_linkage_table(fs_as_ps)

    def local_explanation_of_ps(self, ps: PS):
        local_importances = self.local_importance_metric.get_local_importance_array(ps)
        local_linkages = self.local_linkage_metric.get_local_linkage_table(ps)

        # TODO find a good way to display them

    @staticmethod
    def only_non_obscured_pss(pss: list[PS]) -> list[PS]:
        def obscures(ps_a: PS, ps_b: PS):
            a_fixed_pos = set(ps_a.get_fixed_variable_positions())
            b_fixed_pos = set(ps_b.get_fixed_variable_positions())
            if a_fixed_pos == b_fixed_pos:
                return False
            return b_fixed_pos.issubset(a_fixed_pos)

        def get_those_that_are_not_obscured_by(ps_list: PS, candidates: set[PS]) -> set[PS]:
            return {candidate for candidate in candidates if not obscures(ps_list, candidate)}

        current_candidates = set(pss)

        for ps in pss:
            current_candidates = get_those_that_are_not_obscured_by(ps, current_candidates)

        return list(current_candidates)

    def explanation_loop(self, evaluated_sampled_solutions: list[EvaluatedFS]):
        first_round = True

        while True:
            if first_round:
                print("Would you like to see some explanations of the solutions? Write an index, or n to exit")
            else:
                print("Type another index, or n to exit")
            answer = input()
            if answer.upper() == "N":
                break
            else:
                try:
                    index = int(answer)
                except ValueError:
                    print("That didn't work, please retry")
                    continue
                solution_to_explain = evaluated_sampled_solutions[index]
                self.local_explanation_of_full_solution(solution_to_explain.full_solution)
