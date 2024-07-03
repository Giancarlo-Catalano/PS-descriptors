import random


def truncation_selection(population: list, amount: int) -> list:
    return sorted(population, reverse=True)[:amount]


def tournament_selection(population: list, amount: int, tournament_size=3) -> list:
    def select_one():
        return max(random.choices(population, k=tournament_size))  # isn't this beautiful? God bless comparable objects

    return [select_one() for _ in range(amount)]
