import warnings
from typing import Optional, TypeAlias

from BenchmarkProblems.BT.RotaPattern import RotaPattern
from FirstPaper.custom_types import JSON

Skill: TypeAlias = str

class Worker:
    available_skills: set[Skill]
    available_rotas: list[RotaPattern]
    worker_id: str
    name: str

    def __init__(self,
                 available_skills: set[Skill],
                 available_rotas: list[RotaPattern],
                 worker_id: str,
                 name: str):
        self.available_skills = available_skills
        self.available_rotas = available_rotas
        self.worker_id = worker_id
        self.name = name

        assert (self.are_rotas_valid())

    def __repr__(self):
        return self.name

    def are_rotas_valid(self) -> bool:
        if len(self.available_rotas) < 1:
            warnings.warn("Attempting to construct a worker with no rotas")
            return False

        return True

        # rota_length = len(self.available_rotas[0])
        #
        # rotas_fit_well = all(len(rota) == rota_length for rota in self.available_rotas)
        # if not rotas_fit_well:
        #     warnings.warn("Attempting to construct a rota which is not a multiple of the week length")
        # return rotas_fit_well

    def get_rota_length(self) -> int:
        return len(self.available_rotas[0])

    def get_week_length(self):
        return self.available_rotas[0].workweek_length

    def get_variable_cardinalities(self, custom_starting_days: bool):
        if custom_starting_days:
            return [len(self.available_rotas), self.get_rota_length() // self.get_week_length()]
        else:
            return [len(self.available_rotas)]

    def to_json(self) -> JSON:
        result = dict()
        result["available_skills"] = list(self.available_skills)
        result["available_rotas"] = [rota.to_json() for rota in self.available_rotas]
        result["worker_id"] = self.worker_id
        result["name"] = self.name
        return result

    @classmethod
    def from_json(cls, data: JSON):
        available_skills = set(elem for elem in data["available_skills"])
        available_rotas = [RotaPattern.from_json(elem) for elem in data["available_rotas"]]
        worker_id = data["worker_id"]
        name = data["name"]
        return cls(available_skills=available_skills,
                   available_rotas=available_rotas,
                   worker_id=worker_id,
                   name=name)


class WorkerVariables:
    which_rota: Optional[int]
    starting_week: Optional[int]

    def __init__(self, which_rota: Optional[int], starting_day: Optional[int]):
        self.which_rota = which_rota
        self.starting_week = starting_day

    def effective_rota(self, worker: Worker, consider_starting_week: bool):
        if consider_starting_week:
            return worker.available_rotas[self.which_rota].with_starting_week(starting_week=self.starting_week)
        else:
            return worker.available_rotas[self.which_rota]
