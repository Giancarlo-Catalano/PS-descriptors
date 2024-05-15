from itertools import chain
from operator import attrgetter

import numpy as np
from deap import tools
from deap.tools import sortNondominated, sortLogNondominated, uniform_reference_points, selNSGA3
from deap.tools.emo import find_intercepts, find_extreme_points, associate_to_niche, NSGA3Memory

from Core.PS import STAR

def get_food_supplies(population) -> np.ndarray:
    """The result is an array, where for each variable in the search space we give the proportion
    of the individuals in the population which have that variable fixed"""
    counts = np.sum([individual.values != STAR for individual in population], dtype=float, axis=0)
    return np.divide(1.0, counts, out=np.zeros_like(counts), where=counts != 0)

def get_food_score(individual, fixed_counts_supply: np.ndarray):
    if individual.is_empty():
        return 0 #np.average(fixed_counts_supply)  # very arbitrary to be honest
    return np.average([food for val, food in zip(individual, fixed_counts_supply)
                         if val != STAR])



def gc_select_from_last_front(last_pareto_front, entire_population, amount_to_select: int):
    food_supply = get_food_supplies(entire_population)  #note: on the entire population, not on the front
    for individual in last_pareto_front:
        individual.fitness.crowding_dist = get_food_score(individual, food_supply)
    sorted_front = sorted(last_pareto_front, key=attrgetter("fitness.crowding_dist"), reverse=True)
    return sorted_front[:amount_to_select]


def gc_selNSGA2(individuals, k, nd='standard'):
    """Apply NSGA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *k* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *individuals*. For more
    details on the NSGA-II operator see [Deb2002]_.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    # this is new, read the comments below


    if nd == 'standard':
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    # usually, here you would assign a crowing distance like so:
    # for front in pareto_fronts:
    #  gc_assignCrowdingDist(front)

    # instead, we ignore the fronts and assign our own crowding at the start


    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        selected = gc_select_from_last_front(pareto_fronts[-1], individuals, k)
        chosen.extend(selected)

    return chosen




class GC_selNSGA3WithMemory(object):
    """Class version of NSGA-III selection including memory for best, worst and
    extreme points. Registering this operator in a toolbox is a bit different
    than classical operators, it requires to instantiate the class instead
    of just registering the function::

        >>> from deap import base
        >>> ref_points = uniform_reference_points(nobj=3, p=12)
        >>> toolbox = base.Toolbox()
        >>> toolbox.register("select", GC_selNSGA3WithMemory(ref_points))

    """
    def __init__(self, ref_points, nd="log"):
        self.ref_points = ref_points
        self.nd = nd
        self.best_point = np.full((1, ref_points.shape[1]), np.inf)
        self.worst_point = np.full((1, ref_points.shape[1]), -np.inf)
        self.extreme_points = None

    def __call__(self, individuals, k):
        chosen, memory = gc_selNSGA3(individuals, k, self.ref_points, self.nd,  # this is the important part
                                  self.best_point, self.worst_point,
                                  self.extreme_points, True)
        self.best_point = memory.best_point.reshape((1, -1))
        self.worst_point = memory.worst_point.reshape((1, -1))
        self.extreme_points = memory.extreme_points
        return chosen


def gc_selNSGA3(individuals, k, ref_points, nd="log", best_point=None,
             worst_point=None, extreme_points=None, return_memory=False):
    """Implementation of NSGA-III selection as presented in [Deb2014]_.

    This implementation is partly based on `lmarti/nsgaiii
    <https://github.com/lmarti/nsgaiii>`_. It departs slightly from the
    original implementation in that it does not use memory to keep track
    of ideal and extreme points. This choice has been made to fit the
    functional api of DEAP. For a version of NSGA-III see
    :class:`~deap.tools.selNSGA3WithMemory`.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param ref_points: Reference points to use for niching.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :param best_point: Best_run point found at previous generation. If not provided
        find the best point only from current individuals.
    :param worst_point: Worst point found at previous generation. If not provided
        find the worst point only from current individuals.
    :param extreme_points: Extreme points found at previous generation. If not provided
        find the extreme points only from current individuals.
    :param return_memory: If :data:`True`, return the best, worst and extreme points
        in addition to the chosen individuals.
    :returns: A list of selected individuals.
    :returns: If `return_memory` is :data:`True`, a namedtuple with the
        `best_point`, `worst_point`, and `extreme_points`.


    You can generate the reference points using the :func:`uniform_reference_points`
    function::

        >>> ref_points = tools.uniform_reference_points(nobj=3, p=12)   # doctest: +SKIP
        >>> selected = selNSGA3(population, k, ref_points)              # doctest: +SKIP

    .. [Deb2014] Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
        Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
        Part I: Solving Problems With Box Constraints. IEEE Transactions on
        Evolutionary Computation, 18(4), 577-601. doi:10.1109/TEVC.2013.2281535.
    """
    if nd == "standard":
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == "log":
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception("selNSGA3: The choice of non-dominated sorting "
                        "method '{0}' is invalid.".format(nd))

    # Extract fitnesses as a numpy array in the nd-sort order
    # Use wvalues * -1 to tackle always as a minimization problem
    fitnesses = np.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fitnesses *= -1

    # Get best and worst point of population, contrary to pymoo
    # we don't use memory
    if best_point is not None and worst_point is not None:
        best_point = np.min(np.concatenate((fitnesses, best_point), axis=0), axis=0)
        worst_point = np.max(np.concatenate((fitnesses, worst_point), axis=0), axis=0)
    else:
        best_point = np.min(fitnesses, axis=0)
        worst_point = np.max(fitnesses, axis=0)

    extreme_points = find_extreme_points(fitnesses, best_point, extreme_points)
    front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = associate_to_niche(fitnesses, ref_points, best_point, intercepts)

    # Get counts per niche for individuals in all front but the last
    niche_counts = np.zeros(len(ref_points), dtype=np.int64)
    index, counts = np.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
    niche_counts[index] = counts

    # Choose individuals from all fronts but the last
    chosen = list(chain(*pareto_fronts[:-1]))

    # Use niching to select the remaining individuals
    sel_count = len(chosen)
    # n = k - sel_count

    ##  GC PART
    selected = gc_select_from_last_front(pareto_fronts[-1], individuals, k)
    chosen.extend(selected)

    # selected = niching(pareto_fronts[-1], n, niches[sel_count:], dist[sel_count:], niche_counts)
    # chosen.extend(selected)
    ## end of GC Part

    if return_memory:
        return chosen, NSGA3Memory(best_point, worst_point, extreme_points)
    return chosen