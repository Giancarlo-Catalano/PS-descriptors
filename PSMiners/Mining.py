import os
from typing import Literal

import numpy as np
import pandas as pd

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core import TerminationCriteria
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef, plot_solutions_in_pRef
from Core.PS import PS
from Core.ArchivePSMiner import ArchivePSMiner
from FSStochasticSearch.HistoryPRefs import uniformly_random_distribution_pRef, pRef_from_GA, pRef_from_SA, \
    pRef_from_GA_best, pRef_from_SA_best
from PSMiners.AbstractPSMiner import AbstractPSMiner
from PSMiners.DEAP.DEAPPSMiner import DEAPPSMiner
from PSMiners.DEAP.deap_utils import report_in_order_of_last_metric, plot_stats_for_run
from PSMiners.PyMoo.SequentialCrowdingMiner import SequentialCrowdingMiner
from utils import announce
import plotly.express as px


def get_history_pRef(benchmark_problem: BenchmarkProblem,
                     sample_size: int,
                     which_algorithm: Literal["uniform", "GA", "SA", "GA_best", "SA_best"],
                     verbose=True):
    with announce(f"Running the algorithm to generate the PRef using {which_algorithm}", verbose=verbose):
        match which_algorithm:
            case "uniform": return uniformly_random_distribution_pRef(sample_size=sample_size,
                                                                      benchmark_problem=benchmark_problem)
            case "GA": return pRef_from_GA(benchmark_problem=benchmark_problem,
                                           sample_size=sample_size,
                                           ga_population_size=300)
            case "SA": return pRef_from_SA(benchmark_problem=benchmark_problem,
                                           sample_size=sample_size,
                                           max_trace = sample_size)
            case "GA_best": return pRef_from_GA_best(benchmark_problem=benchmark_problem,
                                                     sample_size=sample_size,
                                                     fs_evaluation_budget=sample_size * 100, # TODO decide elsewhere
                                                     )
            case "SA_best": return pRef_from_SA_best(benchmark_problem=benchmark_problem,
                                                     sample_size=sample_size)
            case _: raise ValueError



def get_ps_miner(pRef: PRef,
                 which: Literal["classic", "NSGA_experimental_crowding", "NSGA", "SPEA2"]):
    match which:
        case "classic": return ArchivePSMiner.with_default_settings(pRef)
        case "NSGA": return DEAPPSMiner(population_size = 300,
                                        uses_custom_crowding = False,
                                        pRef = pRef)
        case "NSGA_experimental_crowding":  return DEAPPSMiner(population_size = 300,
                                                               uses_custom_crowding = True,
                                                               pRef = pRef)
        case "SPEA2": return DEAPPSMiner(population_size = 300,
                                         uses_custom_crowding = True,
                                         pRef = pRef,
                                         use_spea=True)
        case "sequential": return SequentialCrowdingMiner.with_default_settings(pRef)
        case _: raise ValueError

def write_pss_to_file(pss: list[PS], file: str):
    ps_matrix = np.array([ps.values for ps in pss])
    np.savez(file, ps_matrix = ps_matrix)

def write_evaluated_pss_to_file(e_pss: list[EvaluatedPS], file: str):
    ps_matrix = np.array([e_ps.values for e_ps in e_pss])
    fitness_matrix = np.array([e_ps.metric_scores for e_ps in e_pss])

    np.savez(file, ps_matrix = ps_matrix, fitness_matrix=fitness_matrix)

def load_pss(file: str) -> list[[EvaluatedPS | PS]]:
    results_dict = np.load(file)
    ps_matrix = results_dict["ps_matrix"]

    pss = [PS(row) for row in ps_matrix]

    if "fitness_matrix" in results_dict:
        fitness_matrix = results_dict["fitness_matrix"]
        return[EvaluatedPS(ps, metric_scores=list(fitness_values))
                 for ps, fitness_values in zip(pss, fitness_matrix)]
    else:
        return pss






def obtain_pss(benchmark_problem: BenchmarkProblem,
            folder: str,
            pRef_obtainment_method: Literal["uniform", "GA", "SA", "GA_best", "SA_best"] = "SA",
            ps_miner_method : Literal["classic", "NSGA", "NSGA_experimental_crowding"] = "NSGA_experimental_crowding",
            pRef_size: int = 10000,
            ps_budget: int = 10000,
            verbose=False):

    history_pRef_file = os.path.join(folder, "history_pRef.npz")
    result_ps_file = os.path.join(folder, "ps_file.npz")


    pRef = get_history_pRef(benchmark_problem=benchmark_problem,
                            sample_size=pRef_size,
                            which_algorithm=pRef_obtainment_method)

    pRef.save(history_pRef_file, verbose)


    algorithm = get_ps_miner(pRef, which=ps_miner_method)

    with announce(f"Running {algorithm} on {pRef} with {ps_budget =}"):
        termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_limit=ps_budget)
        algorithm.run(termination_criterion, verbose=verbose)

    result_ps = algorithm.get_results(None)
    result_ps = AbstractPSMiner.without_duplicates(result_ps)



    write_evaluated_pss_to_file(result_ps, result_ps_file)

    if verbose:
        report_in_order_of_last_metric(result_ps, benchmark_problem, limit_to=12)

    if verbose and isinstance(algorithm, DEAPPSMiner):
        logbook = algorithm.last_logbook
        nsga_run_plot_name_max = os.path.join(folder, "nsga_plot_max.png")
        nsga_run_plot_name_avg = os.path.join(folder, "nsga_plot_avg.png")
        print(f"Plotting the NSGA run in {nsga_run_plot_name_max}, {nsga_run_plot_name_avg}")
        plot_stats_for_run(logbook, nsga_run_plot_name_max, show_max=True, show_mean=False)
        plot_stats_for_run(logbook, nsga_run_plot_name_avg, show_max=False, show_mean=True)



    if verbose:
        pRef_plot_name = os.path.join(folder, "pRef_plot.png")
        print(f"Plotting the pRef in {pRef_plot_name}")
        plot_solutions_in_pRef(pRef, pRef_plot_name)


def view_3d_plot_of_pss(ps_file: str):

    e_pss = load_pss(ps_file)
    metric_matrix = np.array([e_ps.metric_scores for e_ps in e_pss])
    df = pd.DataFrame(metric_matrix, columns=["Simplicity", "Mean Fitness", "Atomicity"])
    # Create a 3D scatter plot with Plotly Express
    fig = px.scatter_3d(
        df,
        x="Simplicity",
        y="Mean Fitness",
        z="Atomicity",
        title="3D Scatter Plot of Simplicity, Mean Fitness, and Atomicity",
        labels={
            "Simplicity": "Simplicity",
            "Mean Fitness": "Mean Fitness",
            "Atomicity": "Atomicity"
        }
    )

    fig.show()

