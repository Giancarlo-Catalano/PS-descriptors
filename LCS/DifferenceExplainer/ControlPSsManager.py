from typing import Optional

import pandas as pd

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PS import PS
from PSMiners.Mining import load_pss, write_pss_to_file


class ControlPSsManager:
    optimisation_problem: BenchmarkProblem
    control_pss_file: str
    control_descriptors_table_file: str

    control_pss: Optional[list[PS]]
    control_descriptors_table: Optional[pd.DataFrame]

    control_samples_per_size_category: int
    sizes_for_which_control_has_been_generated: Optional[set[int]]

    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 control_pss_file: str,
                 control_descriptors_table_file: str,
                 control_samples_per_size_category: int):
        self.optimisation_problem = optimisation_problem

        self.control_pss_file = control_pss_file
        self.control_descriptors_table_file = control_descriptors_table_file

        self.control_pss = None
        self.control_descriptors_table = None

        self.sizes_for_which_control_has_been_generated = None
        self.control_samples_per_size_category = control_samples_per_size_category

    @property
    def search_space(self):
        return self.optimisation_problem.search_space

    def load_from_files(self):
        self.control_pss = load_pss(self.control_pss_file)
        self.control_descriptors_table = pd.read_csv(self.control_descriptors_table_file)

        self.sizes_for_which_control_has_been_generated = set(self.control_descriptors_table["size"].unique())

    def get_descriptors_of_ps(self, ps: PS) -> dict[str, float]:
        result = self.optimisation_problem.get_descriptors_of_ps(ps)
        result["size"] = ps.fixed_count()
        return result

    def generate_data_for_new_size_category(self, size_category: int):
        """Generates the new control pss and the descriptors.
        It updates the internal control pss, the descriptor table and the 'sizes_for_which_control_has_been_generated"""

        new_control_pss = [PS.random_with_fixed_size(self.search_space, size_category)
                           for _ in range(self.control_samples_per_size_category)]

        new_property_rows = pd.DataFrame(data=[self.get_descriptors_of_ps(ps) for ps in new_control_pss])

        self.control_pss.extend(new_control_pss)
        self.control_descriptors_table = pd.concat([self.control_descriptors_table_file, new_property_rows])
        self.sizes_for_which_control_has_been_generated.add(size_category)

    def start_from_scratch(self):
        self.control_pss = []
        self.control_descriptors_table = pd.DataFrame()
        self.sizes_for_which_control_has_been_generated = set()

    def write_to_files(self):
        write_pss_to_file(pss=self.control_pss, file=self.control_pss_file)
        self.control_descriptors_table.to_csv(self.control_descriptors_table_file)
