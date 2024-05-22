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
                              calendar_length=7*2,
                              rota_preference_weight=0)




def get_two_team_instance():
    " This one works, do not touch!"
    start_of_week_rota = make_rota_pattern_from_string("WWW----")
    end_first_weekend = make_rota_pattern_from_string("---WWW----WW--")
    end_second_weekend = make_rota_pattern_from_string("---WW-----WWW-")

    wrong_rota = make_rota_pattern_from_string("W-W-W--")


    def make_worker(name: str, pattern: RotaPattern, skill: str) -> Worker:
        return Worker(available_skills={skill},
                      available_rotas=[wrong_rota, pattern],
                      worker_id=f"ID_{name}",
                      name=name)


    workers = []
    def add_custom_worker(is_start: bool, offset_weekend: bool =  False, skill: str = "X"):
        if is_start:
            rota = start_of_week_rota
        else:
            rota = end_first_weekend if offset_weekend else end_second_weekend

        name = f"{'START' if is_start else 'END'}"

        workers.append(make_worker(name, rota, skill))



    add_custom_worker(is_start=True, skill="X")
    add_custom_worker(is_start=True, skill="Y")
    add_custom_worker(is_start=True, skill="Z")
    add_custom_worker(is_start=False, offset_weekend=True, skill="X")
    add_custom_worker(is_start=False, offset_weekend=False, skill="Y")
    add_custom_worker(is_start=False, offset_weekend=False, skill="Z")

    for index, worker in enumerate(workers):
        worker.name = worker.name + f"_{index}"


    return EfficientBTProblem(workers = workers,
                              calendar_length=7*2)




def get_stand_end_team_problem(amount_of_skills: int):
    start_of_week_rota = make_rota_pattern_from_string("WWW----")
    end_first_weekend = make_rota_pattern_from_string("---WWW----WW--")
    end_second_weekend = make_rota_pattern_from_string("---WW-----WWW-")

    wrong_rota = make_rota_pattern_from_string("W-W-W--")


    def make_worker(name: str, pattern: RotaPattern, skill: str) -> Worker:
        return Worker(available_skills={skill},
                      available_rotas=[wrong_rota, pattern],
                      worker_id=f"ID_{name}",
                      name=name)


    workers = []
    def add_custom_worker(is_start: bool, offset_weekend: bool =  False, skill: str = "undecided"):
        if is_start:
            rota = start_of_week_rota
        else:
            rota = end_first_weekend if offset_weekend else end_second_weekend

        name = f"{'START' if is_start else 'END'}"

        workers.append(make_worker(name, rota, skill))


    for skill in range(amount_of_skills):
        add_custom_worker(is_start=True, skill=f"{amount_of_skills}")
        add_custom_worker(is_start=False, offset_weekend=True, skill=f"{amount_of_skills}")
        add_custom_worker(is_start=False, offset_weekend=False, skill=f"{amount_of_skills}")

    for index, worker in enumerate(workers):
        worker.name = worker.name + f"_{index}"

    return EfficientBTProblem(workers = workers,
                              calendar_length=7*2)


def get_start_end_team_problem(amount_of_skills: int):
    full_week_alpha = make_rota_pattern_from_string("WWWWWW-WWWWW--")
    full_week_beta = make_rota_pattern_from_string("WWWWW--WWWWWW-")

    lazy_rota_alpha = make_rota_pattern_from_string("W------")
    lazy_rota_beta = make_rota_pattern_from_string("-W-----")


    def make_worker(name: str, patterns: list[RotaPattern], skill: str) -> Worker:
        return Worker(available_skills={skill},
                      available_rotas=patterns,
                      worker_id=f"ID_{name}",
                      name=name)


    workers = []
    for skill in range(amount_of_skills):
        workers.append(make_worker(f"{skill}_alpha", [full_week_alpha, lazy_rota_alpha],
                                   skill=f"SKILL_{skill}"))
        workers.append(make_worker(f"{skill}_beta", [full_week_beta, lazy_rota_beta],
                                   skill=f"SKILL_{skill}"))

    return EfficientBTProblem(workers = workers,
                              calendar_length=7*2)

