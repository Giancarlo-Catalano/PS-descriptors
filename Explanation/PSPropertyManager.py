import itertools
from typing import TypeAlias, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats

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
            properties_dicts = [self.get_sanitised_descriptors_of_ps(ps) for ps in itertools.chain(pss, control_pss)]
            properties_df = pd.DataFrame(properties_dicts)
            properties_df["control"] = np.array([index >= len(pss) for index in range(len(properties_dicts))])   # not my best work
            properties_df["size"] = np.array([ps.fixed_count() for ps in itertools.chain(pss, control_pss)])

            properties_df.to_csv(self.property_table_file, index=False)
        self.cached_property_table = properties_df

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

    def get_sanitised_descriptors_of_ps(self, ps: PS) -> dict[str, float]:
        return {key: float(value)
                for key, value in self.problem.get_descriptors_of_ps(ps).items()}

    def is_property_rank_significant(self, rank: PropertyRank) -> bool:
        is_low = rank < self.threshold
        is_high = 1-rank < self.threshold
        return is_low or is_high
    def get_significant_properties_of_ps(self, ps: PS) -> list[PVR]:
        pvrs = [(name, value, self.get_rank_of_property(ps, name, value))
                for name, value in self.get_sanitised_descriptors_of_ps(ps).items()]
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


    def get_variable_properties_stats(self, pss: list[PS], var_index: int, value: Optional[int] = None) -> dict:

        # TODO think about this more thoroughly
        # should we compare against control PSs or experimental PSs?

        if value is None:
            which_pss_contain_var = [var_index in ps.get_fixed_variable_positions()
                                     for ps in pss]
        else:
            which_pss_contain_var = [ps[var_index] == value
                                     for ps in pss]
        relevant_properties = self.property_table[self.property_table["control"]==False][which_pss_contain_var]
        relevant_properties = relevant_properties[relevant_properties["size"] > 1]
        control_properties = self.property_table[self.property_table["control"]==True]


        def valid_column_values_from(df: pd.DataFrame, column_name):
            """ This is why I hate pandas"""
            column = df[column_name].copy()
            column.dropna(inplace=True)
            column = column[~np.isnan(column)]
            """ this tiny snipped took me half an hour, by the way. Modify with care"""
            return column.values

        def p_value_of_difference_of_means_and_mean(property_name: str) -> (float, float, float):
            experimental_values = valid_column_values_from(relevant_properties, property_name)
            control_values = valid_column_values_from(control_properties, property_name)

            if len(experimental_values) < 2 or len(control_values) < 2:
                return 1.0
            t_value, p_value = stats.ttest_ind(experimental_values, control_values)
            return p_value, np.average(experimental_values), np.average(control_values)

        properties_and_p_values = {prop: p_value_of_difference_of_means_and_mean(prop)
                                   for prop in control_properties.columns
                                   if prop != "size"
                                   if prop != "control"}

        return properties_and_p_values



    def plot_var_property(self, var_index: int, value: Optional[int], property_name: str, pss: list[PS]):
        if value is None:
            which_pss_contain_var = [var_index in ps.get_fixed_variable_positions()
                                     for ps in pss]
        else:
            which_pss_contain_var = [ps[var_index] == value
                                     for ps in pss]

        if not any(which_pss_contain_var):
            print("Could not produce a plot, insufficient pss")

        relevant_properties = self.property_table[self.property_table["control"]==False][which_pss_contain_var]
        relevant_properties = relevant_properties[relevant_properties["size"] > 1]
        control_properties = self.property_table[self.property_table["control"]==True]

        def valid_column_values_from(df: pd.DataFrame, column_name):
            """ This is why I hate pandas"""
            column = df[column_name].copy()
            column.dropna(inplace=True)
            column = column[~np.isnan(column)]
            """ this tiny snipped took me half an hour, by the way. Modify with care"""
            return column.values

        experimental_values = valid_column_values_from(relevant_properties, property_name)
        control_values = valid_column_values_from(control_properties, property_name)

        plt.hist(control_values, bins=20, alpha=0.5, label='Control', color='blue', edgecolor='black', density=True)
        plt.hist(experimental_values, bins=20, alpha=0.5, label='Experimental', color='red', edgecolor='black', density=True)

        # Add labels and title
        plt.xlabel(self.problem.get_readable_property_name(property_name))
        plt.ylabel('Frequency')
        plt.title('Distribution of Control vs Experimental Values')

        # Add legend
        plt.legend()

        # Show plot
        plt.show()



    def get_available_properties(self) -> list[str]:
        return list(self.property_table.columns)

