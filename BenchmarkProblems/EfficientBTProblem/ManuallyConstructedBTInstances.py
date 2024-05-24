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
    alternative_rota = make_rota_pattern_from_string("-------WWWWWW-")
    starting_shift = make_rota_pattern_from_string("WWW----")
    ending_shift = make_rota_pattern_from_string("---WWW-")

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

        workers.append(Worker(available_skills=skills,
                              available_rotas=[alternative_rota, starting_shift],
                              name=f"Starting_{skill_number}",
                              worker_id=random_id()))

        workers.append(Worker(available_skills=skills,
                              available_rotas=[alternative_rota, ending_shift],
                              name=f"Ending_{skill_number}",
                              worker_id=random_id()))


    return EfficientBTProblem(workers=workers, calendar_length=7*12, rota_preference_weight=0)



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
                              calendar_length=7*12,
                              rota_preference_weight=0)



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


    return EfficientBTProblem(workers=workers, calendar_length=7*12, rota_preference_weight=0)




def get_bad_week_instance(amount_of_skills: int = 2, workers_per_skill: int = 2) -> EfficientBTProblem:
    alternative_rota = make_rota_pattern_from_string("-------WWWWWW-")
    starting_shift = make_rota_pattern_from_string("WWW----")
    ending_shift = make_rota_pattern_from_string("---WWW-")

    workers = []
    for skill_number in range(amount_of_skills):
        skills = {f"SKILL_{skill_number}"}
        for _ in range(workers_per_skill):
            workers.append(Worker(available_skills=skills,
                                  available_rotas=[alternative_rota, starting_shift],
                                  name=f"Worker_{skill_number}",
                                  worker_id=random_id()))

    return EfficientBTProblem(workers=workers, calendar_length=7*12, rota_preference_weight=0)



def get_square_instance(amount_of_squares: int) -> EfficientBTProblem:

    def make_worker_aux(rota_1: str, rota_2: str, skills: set[int], name: str) -> Worker:
        rota_1_pattern = make_rota_pattern_from_string(rota_1)
        rota_2_pattern = make_rota_pattern_from_string(rota_2)
        return Worker(available_skills={str(v) for v in skills},
                                  available_rotas=[rota_1_pattern, rota_2_pattern],
                                  name=name,
                                  worker_id=random_id())

    fullw = "WWWWWW-"
    empty = "-------"

    def make_square(initial_skill: int) -> list[Worker]:
        a = initial_skill
        b = initial_skill + 1
        c = initial_skill + 2
        d = initial_skill + 3

        workers = []
        workers.append(make_worker_aux("WWW-----------", fullw+empty, {a, b}, f"AB_{initial_skill}"))
        workers.append(make_worker_aux("-WWW----------", empty+fullw, {b, c}, f"BC_{initial_skill}"))
        workers.append(make_worker_aux("--WWW---------", fullw+empty, {c, d}, f"CD_{initial_skill}"))
        workers.append(make_worker_aux("---WWW--------", empty+fullw, {d, a}, f"DA_{initial_skill}"))

        return workers
    workers = [worker
               for which in range(amount_of_squares)
               for worker in make_square(which*4)]

    return EfficientBTProblem(workers=workers, calendar_length=7*12, rota_preference_weight=0)