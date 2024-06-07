from typing import Iterable, Any

import utils
from Core.FSEvaluator import Fitness


class TerminationCriteria:
    """ This class is used to provide a clean mechanism to control terminate an iterative process """
    """ The main class you'll be using is PSEvaluationLimit"""

    def __init__(self):
        pass

    def __repr__(self):
        raise Exception("Implementation of TerminationCriteria does not implement __repr__")

    def met(self, **kwargs):
        raise Exception("Implementation of TerminationCriteria does not implement termination_criteria_met")


class AsLongAsWanted(TerminationCriteria):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"AsLongAsWanted"

    def met(self, **kwargs):
        return False


class TimeLimit(TerminationCriteria):
    max_time: float

    def __init__(self, max_time: float):
        super().__init__()
        self.max_time = max_time

    def __repr__(self):
        return f"TimeLimit({self.max_time})"

    def met(self, **kwargs):
        return kwargs["time"] >= self.max_time


class IterationLimit(TerminationCriteria):
    max_iterations: float

    def __init__(self, max_iterations: float):
        super().__init__()
        self.max_iterations = max_iterations

    def __repr__(self):
        return f"IterationLimit({self.max_iterations})"

    def met(self, **kwargs):
        return kwargs["iterations"] >= self.max_iterations


class UntilAllTargetsFound(TerminationCriteria):
    targets: Iterable[Any]

    def __init__(self, targets: Iterable[Any]):
        super().__init__()
        self.targets = targets

    def __repr__(self):
        return f"Targets({self.targets})"

    def met(self, **kwargs):
        archive = kwargs["archive"]
        return all(target in archive for target in self.targets)


class UntilGlobalOptimaReached(TerminationCriteria):
    global_optima_fitness: Fitness

    def __init__(self, global_optima_fitness: Fitness):
        super().__init__()
        self.global_optima_fitness = global_optima_fitness

    def __repr__(self):
        return f"UntilGlobalOptima({self.global_optima_fitness})"

    def met(self, **kwargs):
        would_be_returned = kwargs["best_fs_fitness"]

        return would_be_returned >= self.global_optima_fitness


class UnionOfCriteria(TerminationCriteria):
    subcriteria: list[TerminationCriteria]

    def __init__(self, *subcriteria):
        self.subcriteria = list(subcriteria)
        super().__init__()

    def __repr__(self):
        return "Union(" + ", ".join(f"{sc}" for sc in self.subcriteria) + ")"

    def met(self, **kwargs):
        return any(sc.met(**kwargs) for sc in self.subcriteria)


class FullSolutionEvaluationLimit(TerminationCriteria):
    fs_limit: int

    def __init__(self, fs_limit: int):
        self.fs_limit = fs_limit
        super().__init__()

    def __repr__(self):
        return f"FSEvaluationLimit({self.fs_limit}"

    def met(self, **kwargs):
        return kwargs["fs_evaluations"] >= self.fs_limit


class PSEvaluationLimit(TerminationCriteria):
    ps_limit: int

    def __init__(self, ps_limit: int):
        self.ps_limit = ps_limit
        super().__init__()

    def __repr__(self):
        return f"PSEvaluationLimit({self.ps_limit})"

    def met(self, **kwargs):
        return kwargs["ps_evaluations"] >= self.ps_limit


class SearchSpaceIsCovered:
    def __init__(self):
        pass

    def __repr__(self):
        return f"SearchSpaceIsCovered"


    def met(self, **kwargs):
        coverage = kwargs["coverage"]
        return all(x > 0 for x in coverage)


class ArchiveSizeLimit:
    max_archive_size: int
    def __init__(self, max_archive_size: int):
        self.max_archive_size = max_archive_size


    def __repr__(self):
        return f"ArchiveSizeLimit({self.max_archive_size})"


    def met(self, **kwargs):
        return len(kwargs["archive"] > self.max_archive_size)