from typing import Optional

import numpy as np

from Core.custom_types import JSON


class WorkDay:
    working: bool
    start_time: Optional[int]   # ignored for now
    end_time: Optional[int]     # ignored for now


    def __init__(self, working: bool, start_time: Optional[int], end_time: Optional[int]):
        self.working = working
        self.start_time = start_time
        self.end_time = end_time


    @classmethod
    def working_day(cls, start_time, end_time):
        return cls(True, start_time, end_time)

    @classmethod
    def not_working(cls):
        return cls(False, None, None)

    def __repr__(self):
        if self.working:
            return "W"
            #return f"{self.start_time}~{self.end_time}"
        else:
            return "_"

class RotaPattern:
    workweek_length: int
    days: list[WorkDay]

    def __init__(self, workweek_length: int, days: list[WorkDay]):
        if len(days) % workweek_length != 0:
            print(f"The rota {days} is not valid, wrong length ({len(days)}) (% {workweek_length} != 0)")
        assert(len(days) % workweek_length == 0)
        self.workweek_length = workweek_length
        self.days = days


    def __repr__(self):
        split_by_week = [self.days[(which*self.workweek_length):((which+1)*self.workweek_length)]
                          for which in range(len(self.days) // self.workweek_length)]

        def repr_week(week: list[WorkDay]) -> str:
            return "<" + ", ".join(f"{day}" for day in week)+">"

        return ", ".join(map(repr_week, split_by_week))

    def with_starting_day(self, starting_day: int):
        if (starting_day >= len(self.days)):
            raise Exception(f"In rota {self}, the requested starting day {starting_day} exceeds the total length {len(self.days)}")
        assert(starting_day < len(self.days))
        return RotaPattern(self.workweek_length, self.days[starting_day:]+self.days[:starting_day])

    def with_starting_week(self, starting_week: int):
        return self.with_starting_day(starting_week * self.workweek_length)

    def as_bools(self) -> list[bool]:
        return [day.working for day in self.days]

    def working_days_in_calendar(self, calendar_length: int) -> np.ndarray:
        result = []
        as_bools = self.as_bools()
        while len(result) < calendar_length:
            result.extend(as_bools)

        return np.array(result[:calendar_length])


    def __len__(self):
        return len(self.days)

    def to_json(self) -> JSON:
        days_object = "".join("W" if day.working else "-" for day in self.days)
        ## if time was being used, then the days object might contain the times
        return {"week_size": self.workweek_length,
                "days":days_object}

    @classmethod
    def from_json(cls, data: JSON):
        week_size = data["week_size"]
        def dummy_working_day() -> WorkDay:
            return WorkDay.working_day(start_time=900, end_time=1600)

        days_string = data["days"]
        days = [dummy_working_day() if day == "W" else WorkDay.not_working() for day in days_string]
        return cls(workweek_length=week_size, days=days)



    def __eq__(self, other) -> bool:
        def compare_when_same_length(a: RotaPattern, b: RotaPattern):
            return all(day_a.working == day_b.working for day_a, day_b in zip(a.days, b.days))

        if len(self) == len(other):
            return compare_when_same_length(self, other)
        else:
            smaller, bigger = self, other
            if len(self) > len(other):
                smaller, bigger = bigger, smaller

            new_days = []
            while len(new_days) < len(bigger):
                new_days.extend(smaller.days)

            new_days = new_days[:len(bigger)]
            new_smaller = RotaPattern(self.workweek_length, new_days)
            return compare_when_same_length(new_smaller, bigger)


    def __hash__(self):
        tuple_of_bools = tuple(day.working for day in self.days)
        return hash(tuple_of_bools)

    def to_numpy_array(self) -> np.ndarray:
        return np.array([day.working for day in self.days])


def get_workers_present_each_day_of_the_week(rotas: list[RotaPattern], calendar_length: int) -> np.ndarray:
    all_rotas = np.array([rota.working_days_in_calendar(calendar_length) for rota in rotas])
    workers_per_day = np.sum(all_rotas, axis=0, dtype=int)

    return workers_per_day.reshape((-1, 7))




def range_score(min_amount, max_amount):
    if max_amount == 0:
        return 1
    return ((max_amount - min_amount) / max_amount) ** 2


def faulty_range_score(min_amount, max_amount):
    if min_amount == 0:   # NOTE THE DIFFERENCE
        return 0
    return ((max_amount - min_amount) / max_amount) ** 2


def get_range_scores(workers_per_weekday: np.ndarray, use_faulty_range_score = False):
    """assumes that the input is already a matrix with 7 columns"""
    mins = np.min(workers_per_weekday, axis=0)
    maxs = np.max(workers_per_weekday, axis=0)

    if use_faulty_range_score:
        return [faulty_range_score(min_amount, max_amount)
                for min_amount, max_amount in zip(mins, maxs)]
    else:
        return [range_score(min_amount, max_amount)
                for min_amount, max_amount in zip(mins, maxs)]









