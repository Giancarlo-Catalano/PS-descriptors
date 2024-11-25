import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS
from Core.SearchSpace import SearchSpace


class KnapSackProblem(BenchmarkProblem):
    items: list[(str, float, float)]

    weight_limit: float


    def __init__(self,
                 items: list[(str, float, float)],
                 weight_limit: float):
        self.items = items
        self.weight_limit = weight_limit

        super().__init__(search_space=SearchSpace(2 for item in self.items))

    def repr_item(self, item):
        name, weight, price = item
        return f"{name =}, {weight = :.2f}, {price = :.2f}"
    def __repr__(self):
        return f"items = {', '.join(map(self.repr_item, self.items))}, weight_limit = {self.weight_limit}"

    def fitness_function(self, fs: FullSolution) -> float:
        added_items = [item for included, item in zip(fs.values, self.items) if included]
        total_weight = sum(item_weight for item_name, item_weight, item_price in added_items)
        total_price = sum(item_price for item_name, item_weight, item_price in added_items)

        if total_weight > self.weight_limit:
            return 0
        else:
            return total_price


    def repr_ps(self, ps: PS) -> str:
        definite_yes = [item for item, ps_value in zip(self.items, ps.values)
                        if ps_value == 1]

        definite_no = [item for item, ps_value in zip(self.items, ps.values)
                        if ps_value == 0]

        result = ""
        if len(definite_yes) > 0:
            result += "YES:\n"
            result += utils.indent("\n".join(map(self.repr_item, definite_yes)))

        if len(definite_no) > 0:
            result += "\nNO:\n"
            result += utils.indent("\n".join(map(self.repr_item, definite_no)))

        return result