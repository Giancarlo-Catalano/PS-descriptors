import random

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


def random_id() -> str:
    return f"{random.randrange(10000)}"
def get_start_and_end_instance(amount_of_skills: int = 2) -> EfficientBTProblem:
    alternative_rota = make_rota_pattern_from_string("-W--W---W--W--")
    starting_shift = make_rota_pattern_from_string("WWW----WWW----")
    ending_shift = make_rota_pattern_from_string("----WWW----WWW")

    workers = []
    for skill_number in range(amount_of_skills):
        skills = {f"SKILL_{skill_number}"}
        workers.append(Worker(available_skills=skills,
                              available_rotas=[alternative_rota, starting_shift],
                              name=f"Starting_{skill_number}",
                              worker_id=random_id()))

        workers.append(Worker(available_skills=skills,
                              available_rotas=[alternative_rota, ending_shift],
                              name=f"Ending_{skill_number}",
                              worker_id=random_id()))


    return EfficientBTProblem(workers=workers, calendar_length=7*12)



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



def get_unfairness_instance(amount_of_skills: int):
    full_week_alpha = make_rota_pattern_from_string("WWWWWW-WWWWW--")
    full_week_beta = make_rota_pattern_from_string("WWWWW--WWWWWW-")

    lazy_rota_alpha = make_rota_pattern_from_string("W------")
    lazy_rota_beta = make_rota_pattern_from_string("-W-----")


    workers = []
    for skill_number in range(amount_of_skills):
        skills = {f"SKILL_{skill_number}"}
        workers.append(Worker(available_skills=skills,
                              available_rotas=[lazy_rota_alpha, full_week_alpha],
                              name=f"Worker_{skill_number}_alpha",
                              worker_id=random_id()))
        workers.append(Worker(available_skills=skills,
                              available_rotas=[lazy_rota_beta, full_week_beta],
                              name=f"Worker_{skill_number}_beta",
                              worker_id=random_id()))

    return EfficientBTProblem(workers = workers,
                              calendar_length=7*12)



def get_toestepping_instance(amount_of_skills: int):
    starting_shift = make_rota_pattern_from_string("WWW----")
    alternative_for_starting_rota = make_rota_pattern_from_string("--WWW--")
    ending_shift = make_rota_pattern_from_string("----WWW")
    alternative_for_end_rota = make_rota_pattern_from_string("-WWW---")

    workers = []
    for skill_number in range(amount_of_skills):
        skills = {f"SKILL_{skill_number}"}
        workers.append(Worker(available_skills=skills,
                              available_rotas=[alternative_for_starting_rota, starting_shift],
                              name=f"Starting_{skill_number}",
                              worker_id=random_id()))

        workers.append(Worker(available_skills=skills,
                              available_rotas=[alternative_for_end_rota, ending_shift],
                              name=f"Ending_{skill_number}",
                              worker_id=random_id()))


    return EfficientBTProblem(workers=workers, calendar_length=7*12)


