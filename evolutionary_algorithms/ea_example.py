"""
Example script to run the api. This finds a minimum of a function of two variables
"""

import math
import random

import numpy as np
from ea_api import Individual, mutate, remove_non_fit, reproduce_all, sort_individuals


def fct(input_data: list):
    """
    The fitness function to minimize
    """
    return math.sin(input_data[0]) * math.cos(input_data[1])


def reproduction(ind_a: Individual, ind_b: Individual):
    """
    Reproduction function for this problem
    """
    new_data = np.divide(ind_a.data + ind_b.data, 2)
    return Individual(new_data, ind_a.fitness_func, ind_a.reproduction_alg)


def main():
    """
    Main method
    """
    pop_size = 10
    nbr_epochs = 50
    birth_rate = 0.2
    all_individuals = []
    for _ in range(pop_size):
        data = np.multiply([random.random(), random.random()], 4)
        all_individuals.append(Individual(data, fct, reproduction))
    sort_individuals(all_individuals)

    for i in range(nbr_epochs):
        print(f"Epoch: {i}")
        sort_individuals(all_individuals)
        reproduce_all(all_individuals, birth_rate)
        remove_non_fit(all_individuals, pop_size)
        mutate()

    sort_individuals(all_individuals)
    best_individual = all_individuals[0]
    print(
        f"Minimum point: {best_individual.data}, with value {best_individual.get_fitness()}"
    )


if __name__ == "__main__":
    main()
