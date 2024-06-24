import os
from typing import Literal

import numpy as np
import pandas as pd

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FirstPaper import TerminationCriteria
from FirstPaper.EvaluatedPS import EvaluatedPS
from FirstPaper.PRef import PRef, plot_solutions_in_pRef
from FirstPaper.PS import PS
from FirstPaper.ArchivePSMiner import ArchivePSMiner
from FSStochasticSearch.HistoryPRefs import uniformly_random_distribution_pRef, pRef_from_GA, pRef_from_SA, \
    pRef_from_GA_best, pRef_from_SA_best
from PSMiners.AbstractPSMiner import AbstractPSMiner
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

