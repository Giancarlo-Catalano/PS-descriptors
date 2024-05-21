import itertools
from typing import TypeAlias, Optional

import numpy as np
import pandas as pd

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PS import PS
from utils import announce

PropertyName: TypeAlias = str
PropertyValue: TypeAlias = float
PropertyRank: TypeAlias = float

PVR: TypeAlias = (str, float, float) # stands for property name, property value, significance

class PSPropertyManager:
    problem: BenchmarkProblem
    property_table_file: str
    cached_property_table: Optional[pd.DataFrame]
    verbose: bool
    threshold: float

    def __init__(self,
                 problem: BenchmarkProblem,
                 property_table_file: str,
                 verbose: bool = False,
                 threshold: float = 0.1):
        self.problem = problem
        self.property_table_file = property_table_file
        self.cached_property_table = None
        self.verbose = verbose
        self.threshold = threshold


    @property
    def property_table(self) -> pd.DataFrame:
        if self.cached_property_table is None:
            with announce(f"Reading the property file {self.property_table_file}"):
                self.cached_property_table = pd.read_csv(self.property_table_file)
        return self.cached_property_table


    def generate_property_table_file(self, pss: list[PS], control_pss: list[PS]):
        with announce(f"Generating the properties file and storing it at {self.property_table_file}", self.verbose):
            properties_dicts = [self.problem.ps_to_properties(ps) for ps in itertools.chain(pss, control_pss)]
            properties_df = pd.DataFrame(properties_dicts)
            properties_df["control"] = np.array([index >= len(pss) for index in range(len(properties_dicts))])   # not my best work
            properties_df["size"] = np.array([ps.fixed_count() for ps in itertools.chain(pss, control_pss)])

            properties_df.to_csv(self.property_table_file, index=False)
        self.cached_properties = properties_df

    @classmethod
    def is_useful_property(cls, property_name: PropertyName):
        return (property_name != "control") and (property_name != "size")


    def get_rank_of_property(self, ps: PS, property_name: PropertyName, property_value: PropertyValue) -> PropertyRank:
        order_of_ps = ps.fixed_count()
        is_control = self.property_table["control"] == True
        is_same_size = self.property_table["size"] == order_of_ps
        control_rows = self.property_table[is_control & is_same_size]
        control_values = control_rows[property_name]
        control_values = [value for value in control_values if not np.isnan(value)]

        return utils.ecdf(property_value, control_values)

    def is_property_rank_significant(self, rank: PropertyRank) -> bool:
        is_low = rank < self.threshold
        is_high = 1-rank < self.threshold
        return is_low or is_high
    def get_significant_properties_of_ps(self, ps: PS) -> list[PVR]:
        pvrs = [(name, value, self.get_rank_of_property(ps, name, value))
                 for name, value in self.problem.ps_to_properties(ps).items()]
        pvrs = [(name, value, rank) for name, value, rank in pvrs
                if self.is_property_rank_significant(rank)]

        return pvrs

    def sort_pvrs_by_rank(self, pvrs: list[PVR]):
        def closeness_to_edge(pvr):
            rank = pvr[2]
            return min(rank, 1-rank)
        return sorted(pvrs, key=closeness_to_edge)


    def sort_pss_by_quantity_of_properties(self, pss: list[(PS, list[PVR])]) -> list[PVR]:
        return sorted(pss, reverse=True, key = lambda x: len(x[1]))