from typing import Optional

from BenchmarkProblems.BT.RotaPattern import RotaPattern
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager
from PairExplanation.PairExplanationTester import PairExplanationTester


class BTProblemPrettyPrinter:
    problem: EfficientBTProblem
    descriptor_manager: DescriptorsManager
    pair_finder: PairExplanationTester

    all_rotas_list: list[RotaPattern]
    all_skills_list: list[str]

    @property
    def pRef(self):
        return self.pair_finder.pRef

    def __init__(self,
                 problem: EfficientBTProblem,
                 descriptor_manager: Optional[DescriptorsManager],
                 pair_finder: PairExplanationTester):
        self.problem = problem
        self.pair_finder = pair_finder  # note that this is done first so that the self.pRef property works

        if descriptor_manager is None:
            descriptor_manager = DescriptorsManager.get_empty_descriptor_manager(problem, self.pRef)
        self.descriptor_manager = descriptor_manager

        self.all_rotas_list = self.get_all_rotas_list()
        self.all_skills_list = list(self.problem.all_skills)

    @classmethod
    def get_all_rotas_list(cls, problem: EfficientBTProblem) -> list[RotaPattern]:
        all_rotas = {rota
                     for worker in problem.workers
                     for rota in worker.available_rotas}

        simplified_rotas = list(map(cls.simplify_rota, all_rotas))
        return simplified_rotas

    @classmethod
    def simplify_rota(cls, rota: RotaPattern) -> RotaPattern:
        """ The pattern [WWW---- WWW----] can be simplified into [WWW----] """

        """There's absolutely a better way to do this but I'm worried of not handling edge cases, so here's a plain approach"""
        def cut_rota(rota: RotaPattern, amount_of_weeks: int) -> RotaPattern:
            days = rota.days[:(7*amount_of_weeks)]
            return RotaPattern(workweek_length = 7, days = days)


        original_rota_weeks = len(rota.days) // 7

        replacement_rota = rota
        for size_of_rota in range(7, original_rota_weeks*7, 7):
            smaller_rota = cut_rota(rota, size_of_rota)
            if smaller_rota == rota:
                replacement_rota = smaller_rota
                break

        return replacement_rota


    def get_index_of_rota(self, rota: RotaPattern) -> int:
        return self.all_rotas_list.index(rota)






