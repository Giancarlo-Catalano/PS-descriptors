from typing import Optional

import numpy as np

from Core.PRef import PRef
from Core.PSMetric.Linkage.OutdatedLinkage import MutualInformation
from utils import announce


class MutualInformationManager:
    linkage_table_file: str
    cached_mutual_information: Optional[MutualInformation]
    verbose: bool


    def __init__(self, linkage_table_file: str,
                 cached_mutual_information: Optional[MutualInformation] = None,
                 verbose: bool = False):
        self.linkage_table_file = linkage_table_file
        self.cached_mutual_information = cached_mutual_information
        self.verbose = verbose


    def generate_linkage_table(self, pRef: PRef):
        metric = MutualInformation()
        with announce(f"Generating the linkage table for {pRef}", self.verbose):
            metric.set_pRef(pRef)
        self.cached_mutual_information = metric

        np.savez(self.linkage_table_file, linkage_table = metric.linkage_table)



    @property
    def mutual_information_metric(self) -> MutualInformation:
        if self.cached_mutual_information is None:
            with announce(f"Loading the linkage table from {self.linkage_table_file}"):
                data = np.load(self.linkage_table_file)
                table = data["linkage_table"]
                self.cached_mutual_information = MutualInformation()
                self.cached_mutual_information.linkage_table = table
        return self.cached_mutual_information