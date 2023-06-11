"""
Basic API for performing all essential steps of evolutionary algorithms
"""


class Individual:
    """
    Object of an individual to be used in EA
    """

    def __init__(self, data: list, fitness_func, reproduction_alg) -> None:
        self.data = data
        self.fitness_func = fitness_func
        self.reproduction_alg = reproduction_alg

    def get_fitness(self):
        """
        Returns the fitness of this individual
        """
        return self.fitness_func(self.data)

    def make_offspring(self, partner):
        """
        Makes an offspring between self Individual and partner Individual
        """
        return self.reproduction_alg(self, partner)


def sort_individuals(all_individuals: list):
    """
    Sorts all individuals by their respective fitness
    """
    all_individuals.sort(key=lambda x: x.get_fitness())


def remove_non_fit(all_individuals: list, pop_size):
    """
    Cuts away the unfit individuals, survival of the fittest happens here...
    """
    sort_individuals(all_individuals)
    return all_individuals[:pop_size]


def mutate():
    pass


def reproduce_all(all_individuals: list, birth_rate):
    """
    Reproduces offspring from all parents deemed fit
    """
    sort_individuals(all_individuals)
    max_new_offspring = int(len(all_individuals) * birth_rate)
    new_offsprings = 0
    for idx, individual in enumerate(all_individuals):
        offspring = individual.make_offspring(all_individuals[idx + 1])
        all_individuals.append(offspring)
        new_offsprings += 1
        if max_new_offspring == new_offsprings:
            break
