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
    control_pss_file: str
    control_descriptors_table_file: str

    control_pss: Optional[list[PS]]
    control_descriptors_table: Optional[pd.DataFrame]

    control_samples_per_size_category: int
    sizes_for_which_control_has_been_generated: Optional[set[int]]

    pRef_manager: PRefManager

    speciality_threshold: float

    verbose: bool

    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 control_pss_file: str,
                 control_descriptors_table_file: str,
                 control_samples_per_size_category: int,
                 pRef_manager: PRefManager,
                 speciality_threshold: float = 0.1,
                 verbose: bool = False):
        self.optimisation_problem = optimisation_problem

        self.control_pss_file = control_pss_file
        self.control_descriptors_table_file = control_descriptors_table_file

        self.control_pss = None
        self.control_descriptors_table = None

        self.sizes_for_which_control_has_been_generated = None
        self.control_samples_per_size_category = control_samples_per_size_category

        self.pRef_manager = pRef_manager

        self.speciality_threshold = speciality_threshold

        self.verbose = verbose


    def load_from_existing_if_possible(self):
        control_ps_file_exists = file_exists(self.control_pss_file)
        ps_descriptor_table_file_exists = file_exists(self.control_descriptors_table_file)
        if control_ps_file_exists and ps_descriptor_table_file_exists:
            self.load_from_files()
        else:
            if control_ps_file_exists != ps_descriptor_table_file_exists:
                raise Exception("Only one of the files for the control data is present!")

            if self.verbose:
                print(f"Since no descriptor files were found, the model will be initialised as empty")
            self.start_from_scratch()
            # otherwise nothing needs to happen, the class initialised in a valid empty state

    @property
    def search_space(self):
        return self.optimisation_problem.search_space

    def load_from_files(self):
        if self.verbose:
            print(f"Loading ps control data from {self.control_pss_file} and {self.control_descriptors_table_file}")

        self.control_pss = load_pss(self.control_pss_file)
        self.control_descriptors_table = pd.read_csv(self.control_descriptors_table_file)

        if "size" in self.control_descriptors_table.columns:  # in the cases where the table file was created but the table was still empty.
            self.sizes_for_which_control_has_been_generated = set(self.control_descriptors_table["size"].unique())
        else:
            self.sizes_for_which_control_has_been_generated = set()


    def get_fitness_delta(self, ps: PS) -> float:
        avg_when_present, avg_when_absent = self.pRef_manager.get_average_when_present_and_absent(ps)
        return avg_when_present - avg_when_absent

    def get_descriptors_of_ps(self, ps: PS) -> dict[str, float]:
        result = self.optimisation_problem.get_descriptors_of_ps(ps)
        result["size"] = ps.fixed_count()
        result["delta"] = self.get_fitness_delta(ps)
        return result

    def generate_data_for_new_size_category(self, size_category: int) -> pd.DataFrame:
        """Generates the new control pss and the descriptors.
        It updates the internal control pss, the descriptor table and the 'sizes_for_which_control_has_been_generated,
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

    def write_to_files(self):
        if self.verbose:
            print(f"Storing the data in the files {self.control_pss_file}, {self.control_descriptors_table_file}")
        write_pss_to_file(pss=self.control_pss, file=self.control_pss_file)
        self.control_descriptors_table.to_csv(self.control_descriptors_table_file)


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


    @classmethod
    def get_empty_descriptor_manager(cls, problem: BenchmarkProblem, pRef: PRef):
        pRef_manager = PRefManager(problem=problem,
                                   pRef_file=None,
                                   instantiate_own_evaluator=False,
                                   verbose=True)
        pRef_manager.set_pRef(pRef)

        descriptors_manager = DescriptorsManager(optimisation_problem=problem,
                                                 control_pss_file=None,
                                                 control_descriptors_table_file=None,
                                                 control_samples_per_size_category=1000,
                                                 pRef_manager=pRef_manager,
                                                 verbose=True)

        descriptors_manager.start_from_scratch()
        return descriptors_manager










