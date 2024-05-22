from BenchmarkProblems.BT.RotaPattern import RotaPattern, WorkDay
from BenchmarkProblems.BT.Worker import Worker
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem


def make_rota_pattern_from_string(input: str) -> RotaPattern:
    """ The input string is in the form W--W---"""
    def workday():
        return WorkDay.working_day(900, 500)
    def not_workday():
        return WorkDay.not_working()

    days = [workday() if letter == "W" else not_workday() for letter in input]
    return RotaPattern(workweek_length=7, days=days)


def get_three_team_instance():
    start_of_week_rota = make_rota_pattern_from_string("WW-----")
    mid_week_rota = make_rota_pattern_from_string("--WW---")
    end_first_weekend = make_rota_pattern_from_string("---WWW----WW--")
    end_second_weekend = make_rota_pattern_from_string("---WW-----WWW-")

    wrong_rota = make_rota_pattern_from_string("W-W-W--")


    def make_worker(name: str, pattern: RotaPattern, skill: str) -> Worker:
        return Worker(available_skills={skill},
                      available_rotas=[wrong_rota, pattern],
                      worker_id=f"ID_{name}",
                      name=name)


    start_A = make_worker("START_A", start_of_week_rota, "Skill_X")
    start_B = make_worker("START_B", start_of_week_rota, "Skill_Y")

    mid_C = make_worker("MID_C", mid_week_rota, "Skill_X")
    mid_D= make_worker("MID_D", mid_week_rota, "Skill_Y")

    end_E = make_worker("END_E", end_first_weekend, "Skill_X")
    end_F = make_worker("END_F", end_second_weekend, "Skill_Y")
    end_G = make_worker("END_G", end_first_weekend, "Skill_X")
    end_H = make_worker("END_H", end_second_weekend, "Skill_Y")

    workers = [start_A, start_B, mid_C, mid_D, end_E, end_F, end_G, end_H]

    return EfficientBTProblem(workers = workers,
                              calendar_length=7*2)

