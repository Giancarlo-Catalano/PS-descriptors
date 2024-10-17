from typing import Optional

import utils
from BenchmarkProblems.BT.RotaPattern import RotaPattern
from BenchmarkProblems.BT.Worker import Worker
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
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

        self.all_rotas_list = self.get_all_rotas_list(self.problem)
        self.all_skills_list = sorted(list(self.problem.all_skills))

    @classmethod
    def get_all_rotas_list(cls, problem: EfficientBTProblem) -> list[RotaPattern]:
        # note that two rotas which are equivalent might have different hash values,
        # so simply making a set of them is not guaranteed to remove duplicates
        # if we force the patterns to be in their minimal form, it should work!
        return list({cls.simplify_rota(rota)
                     for worker in problem.workers
                     for rota in worker.available_rotas})

    @classmethod
    def simplify_rota(cls, rota: RotaPattern) -> RotaPattern:
        """ The pattern [WWW---- WWW----] can be simplified into [WWW----] """

        """There's absolutely a better way to do this but I'm worried of not handling edge cases, so here's a plain approach"""
        def cut_rota(rota: RotaPattern, amount_of_weeks: int) -> RotaPattern:
            days = rota.days[:(7*amount_of_weeks)]
            return RotaPattern(workweek_length = 7, days = days)

        if (rota.days[:7] == rota.days[7:]):
            print("poop")

        original_rota_weeks = len(rota.days) // 7

        replacement_rota = rota
        for size_of_rota in range(1, original_rota_weeks):
            smaller_rota = cut_rota(rota, size_of_rota)
            if smaller_rota == rota:
                replacement_rota = smaller_rota
                break

        return replacement_rota


    def get_index_of_rota(self, rota: RotaPattern) -> int:
        return self.all_rotas_list.index(rota)

    def repr_rota_choice(self, rota_index: int) -> str:
        return utils.alphabet[rota_index]

    def repr_rota_index(self, rota: RotaPattern) -> str:
        index = self.all_rotas_list.index(rota)
        return f"ROTA {index+1}"

    def repr_rota(self, rota: RotaPattern) -> str:
        return "\t".join(f"{day}" for day in rota.days)

    def repr_extended_rota(self, rota: RotaPattern) -> str:
        new_days = []
        while len(new_days) < self.problem.calendar_length:
            new_days.extend(rota.days)

        new_days = new_days[:self.problem.calendar_length]
        new_rota = RotaPattern(workweek_length=rota.workweek_length, days = new_days)
        return self.repr_rota(new_rota)

    def repr_skillset(self, skillset: set[str]) -> str:
        return "\t".join(("" if skill in skillset else skill) for skill in self.all_skills_list)

    def repr_worker(self, worker: Worker) -> str:
        skills_str = self.repr_skillset(worker.available_skills)
        rotas_str = "\t".join(self.repr_rota_index(rota) for rota in worker.available_rotas)
        return "\t".join([worker.name, skills_str, rotas_str])



    def repr_problem_workers(self) -> str:
        return "\n".join(map(self.repr_worker, self.problem.workers))

    def repr_problem_rotas(self) -> str:
        return "\n".join("\t".join([self.repr_rota_index(rota), self.repr_rota(rota)])
                         for rota in self.all_rotas_list)

    def repr_partial_solution(self, ps: PS) -> str:
        workers_and_choices = [(worker, choice)
                               for worker, choice in zip(self.problem.workers, ps.values)
                               if choice != STAR]

        def repr_assigned_worker(worker: Worker, choice: int) -> str:
            skillset_str = self.repr_skillset(worker.available_skills)
            rota_choice_label = self.repr_rota_choice(choice)
            actual_rota_str = self.repr_extended_rota(worker.available_rotas[choice])

            return "\t".join([worker.name, rota_choice_label, skillset_str, actual_rota_str])

        return "\n".join([repr_assigned_worker(w, c) for w, c in workers_and_choices])


    def repr_full_solution(self, fs: FullSolution) -> str:
        return self.repr_partial_solution(PS.from_FS(fs))







