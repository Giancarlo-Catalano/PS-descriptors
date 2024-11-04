import json
import os
from typing import Optional

import pandas as pd
from pandas.io.common import file_exists

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PRef import PRef
from Core.PS import PS
from Explanation.PRefManager import PRefManager
from LCS.PSEvaluator import GeneralPSEvaluator
from PSMiners.Mining import load_pss, write_pss_to_file


class DescriptorsManager:
    optimisation_problem: BenchmarkProblem
    directory: str

    control_pss: Optional[list[PS]]
    control_descriptors_table: Optional[pd.DataFrame]

    control_samples_per_size_category: int
    sizes_for_which_control_has_been_generated: Optional[set[int]]

    speciality_threshold: float
    verbose: bool

    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 control_pss: Optional[list[PS]],
                 control_descriptors_table: Optional[pd.DataFrame],
                 control_samples_per_size_category: int,
                 sizes_for_which_control_has_been_generated: Optional[set[int]],
                 speciality_threshold: float = 0.1,
                 verbose: bool = False):
        self.optimisation_problem = optimisation_problem
        self.control_pss = control_pss
        self.control_descriptors_table = control_descriptors_table
        self.control_samples_per_size_category = control_samples_per_size_category
        self.sizes_for_which_control_has_been_generated = sizes_for_which_control_has_been_generated
        self.speciality_threshold = speciality_threshold
        self.verbose = verbose

    @classmethod
    def get_names_of_files(cls, directory: str) -> (str, str, str):
        control_pss_file = os.path.join(directory, "control_pss.npz")
        control_descriptors_table_file = os.path.join(directory, "control_descriptors_table_file")
        other_settings_json_file = os.path.join(directory, "other_settings.json")
        return control_pss_file, control_descriptors_table_file, other_settings_json_file

    @classmethod
    def load(cls,
             problem: BenchmarkProblem,
             directory: str,
             verbose: bool = False):
        control_pss_file, control_descriptors_table_file, other_settings_json_file = cls.get_names_of_files(directory)

        with open(other_settings_json_file, "w") as json_file:
            json_data = json.load(json_file)

        control_samples_per_size_category = json_data["control_samples_per_size_category"]
        speciality_threshold = json_data["speciality_threshold"]
        control_size_categories = set(json_data["control_size_categories"])

        if not file_exists(control_pss_file):
            raise Exception(f"Could not load the control pss file for the descriptors ({control_pss_file})")
        if not file_exists(control_descriptors_table_file):
            raise Exception(
                f"Could not load control_descriptors_table_file for the descriptors ({control_descriptors_table_file})")

        control_pss = load_pss(control_pss_file)
        control_descriptors_table = pd.read_csv(control_descriptors_table_file)

        return cls(optimisation_problem=problem,
                   control_samples_per_size_category=control_samples_per_size_category,
                   speciality_threshold=speciality_threshold,
                   control_descriptors_table=control_descriptors_table,
                   control_pss=control_pss,
                   sizes_for_which_control_has_been_generated=control_size_categories,
                   verbose=verbose)

    def store(self, directory: str):
        control_pss_file, control_descriptors_table_file, other_settings_json_file = self.get_names_of_files(directory)
        write_pss_to_file(pss=self.control_pss, file=control_pss_file)
        self.control_descriptors_table.to_csv(control_descriptors_table_file)

        other_settings_json_contents = {"control_samples_per_size_category": self.control_samples_per_size_category,
                                        "speciality_threshold": self.speciality_threshold,
                                        "control_size_categories": list(self.sizes_for_which_control_has_been_generated)}
        with utils.open_and_make_directories(other_settings_json_file) as json_file:
            json.dump(other_settings_json_contents, json_file, indent=4)

    @classmethod
    def with_no_samples_yet(cls, problem: BenchmarkProblem,
                            specialty_threshold: float,
                            control_samples_per_size_category: int,
                            verbose: bool):
        return cls(optimisation_problem=problem,
                   control_samples_per_size_category=control_samples_per_size_category,
                   speciality_threshold=specialty_threshold,
                   control_descriptors_table=pd.DataFrame(),
                   control_pss=[],
                   sizes_for_which_control_has_been_generated=set(),
                   verbose=verbose)

    @property
    def search_space(self):
        return self.optimisation_problem.search_space

    def get_fitness_delta(self, ps: PS, pRef: PRef) -> float:
        avg_when_present, avg_when_absent = PRefManager.get_average_when_present_and_absent(ps, pRef)
        return avg_when_present - avg_when_absent

    def get_descriptors_of_ps(self, ps: PS) -> dict[str, float]:
        result = self.optimisation_problem.get_descriptors_of_ps(ps)
        result["size"] = ps.fixed_count()
        return result

    def generate_data_for_new_size_category(self, size_category: int) -> pd.DataFrame:
        """Generates the new control pss and the descriptors.
        It updates the internal control pss, the descriptors table and the 'sizes_for_which_control_has_been_generated,
        and returns the new rows generated"""

        if self.verbose:
            print(f"Generating control data for size category = {size_category}")

        new_control_pss = [PS.random_with_fixed_size(self.search_space, size_category)
                           for _ in range(self.control_samples_per_size_category)]

        new_property_rows = pd.DataFrame(data=[self.get_descriptors_of_ps(ps) for ps in new_control_pss])

        self.control_pss.extend(new_control_pss)
        self.control_descriptors_table = pd.concat([self.control_descriptors_table, new_property_rows])
        self.sizes_for_which_control_has_been_generated.add(size_category)
        return new_property_rows

    def start_from_scratch(self):
        self.control_pss = []
        self.control_descriptors_table = pd.DataFrame()
        self.sizes_for_which_control_has_been_generated = set()



    def get_table_rows_where_size_is(self, size: int) -> pd.DataFrame:
        if not size in self.sizes_for_which_control_has_been_generated:
            return self.generate_data_for_new_size_category(size_category=size)
        else:
            return self.control_descriptors_table[self.control_descriptors_table["size"] == size]

    def get_percentiles_for_descriptors(self, ps_size: int, ps_descriptors: dict[str, float]) -> dict[str, float]:
        table_rows = self.get_table_rows_where_size_is(ps_size)

        def get_percentile_of_descriptor(descriptor_name: str) -> float:
            descriptor_value = ps_descriptors[descriptor_name]
            return utils.ecdf(descriptor_value, list(table_rows[descriptor_name]))

        return {descriptor_name: get_percentile_of_descriptor(descriptor_name)
                for descriptor_name in ps_descriptors
                if descriptor_name != "size"}

    def get_significant_descriptors_of_ps(self, ps: PS) -> list[(str, float, float)]:
        descriptors = self.get_descriptors_of_ps(ps)
        size = ps.fixed_count()
        percentiles = self.get_percentiles_for_descriptors(ps_size=size, ps_descriptors=descriptors)

        names_values_percentiles = [(name, descriptors[name], percentiles[name]) for name in percentiles]

        # then we only consider values which are worth reporting
        def percentile_is_significant(percentile: float) -> bool:
            return (percentile < self.speciality_threshold) or (percentile > (1 - self.speciality_threshold))

        # only keep significant descriptors
        names_values_percentiles = [(name, value, percentile)
                                    for name, value, percentile in names_values_percentiles
                                    if percentile_is_significant(percentile) or name == "delta"]

        # sort by "extremeness"
        names_values_percentiles.sort(key=lambda x: abs(0.5 - x[2]), reverse=True)

        return names_values_percentiles

    def descriptors_tuples_into_string(self, names_values_percentiles: list[(str, float, float)], ps: PS) -> str:
        return "\n".join(self.optimisation_problem.repr_property(name, value, percentile, ps)
                         for name, value, percentile in names_values_percentiles)

    def get_descriptors_string(self, ps: PS) -> str:
        names_values_percentiles = self.get_significant_descriptors_of_ps(ps)
        return self.descriptors_tuples_into_string(names_values_percentiles, ps)
