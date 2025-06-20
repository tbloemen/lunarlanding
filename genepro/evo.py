from typing import Callable

import numpy as np
from numpy.random import random as randu
from numpy.random import randint as randi
from numpy.random import choice as randc
from numpy.random import shuffle
import time, inspect
from copy import deepcopy
from joblib.parallel import Parallel, delayed
from typing import List
from functools import cache

from genepro.node import Node
from genepro.variation import *
from genepro.selection import tournament_selection
from compare_expressions import *

import numpy as np


class Individual:
    def __init__(self, objectives=None, reference=None):
        if objectives is not None:
            self.objectives = np.array(objectives)
        else:
            self.objectives = np.zeros(2)

        self.reference = reference
        self.rank = None


class Evolution:
    """
    Class concerning the overall evolution process.

    Parameters
    ----------
    fitness_function : function
      the function used to evaluate the quality of evolving trees, should take a Node and return a float; higher fitness is better

    internal_nodes : list
      list of Node objects to be used as internal nodes for the trees (e.g., [Plus(), Minus(), ...])

    leaf_nodes : list
      list of Node objects to be used as leaf nodes for the trees (e.g., [Feature(0), Feature(1), Constant(), ...])

    pop_size : int, optional
      the population size (default is 256)

    init_max_depth : int, optional
      the maximal depth trees can have at initialization (default is 4)

    max_tree_size : int, optional
      the maximal number of nodes trees can have during the entire evolution (default is 64)

    crossovers : list, optional
      list of dictionaries that contain: "fun": crossover functions to be called, "rate": rate of applying crossover, "kwargs" (optional): kwargs for the chosen crossover function (default is [{"fun":subtree_crossover, "rate": 0.75}])

    mutations : list, optional
      similar to `crossovers`, but for mutation (default is [{"fun":subtree_mutation, "rate": 0.75}])

    coeff_opts : list, optional
      similar to `crossovers`, but for coefficient optimization (default is [{"fun":coeff_mutation, "rate": 1.0}])

    selection : dict, optional
      dictionary that contains: "fun": function to be used to select promising parents, "kwargs": kwargs for the chosen selection function (default is {"fun":tournament_selection,"kwargs":{"tournament_size":4}})

    max_evals : int, optional
      termination criterion based on a maximum number of fitness function evaluations being reached (default is None)

    max_gens : int, optional
      termination criterion based on a maximum number of generations being reached (default is 100)

    max_time: int, optional
      termination criterion based on a maximum runtime being reached (default is None)

    n_jobs : int, optional
      number of jobs to use for parallelism (default is 4)

    verbose : bool, optional
      whether to log information during the evolution (default is False)

    Attributes
    ----------
    All of the parameters, plus the following:

    population : list
      list of Node objects that are the root of the trees being evolved

    num_gens : int
      number of generations

    num_evals : int
      number of evaluations

    start_time : time
      start time

    elapsed_time : time
      elapsed time

    best_of_gens : list
      list containing the best-found tree in each generation; note that the entry at index 0 is the best at initialization
    """

    def __init__(
        self,
        # required settings
        fitness_function: Callable[[Node], float],
        internal_nodes: list,
        leaf_nodes: list,
        # optional evolution settings
        n_trees: int,
        pop_size: int = 256,
        init_max_depth: int = 4,
        max_tree_size: int = 64,
        crossovers: list = [{"fun": subtree_crossover, "rate": 0.5}],
        mutations: list = [{"fun": subtree_mutation, "rate": 0.5}],
        coeff_opts: list = [{"fun": coeff_mutation, "rate": 0.5}],
        selection: dict = {
            "fun": tournament_selection,
            "kwargs": {"tournament_size": 8},
        },
        # termination criteria
        max_evals: int = None,
        max_gens: int = 100,
        max_time: int = None,
        # other
        n_jobs: int = 4,
        verbose: bool = False,
    ):
        self.num_gens = 0
        # const = 0
        # for i in range(50):
        #     const += np.power(np.e, -i*coeff_opts[0]["k"])
        # const = 12.5/const
        # coeff_opts = [{
        #     "fun": coeff_mutation,
        #     "rate": coeff_opts[0]["rate"],
        #     "kwargs": {
        #         "temp": coeff_opts[0]["k"],
        #         "generation": self.num_gens,
        #         "const": const
        #     }
        # }
        # set parameters as attributes
        _, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        # fill-in empty kwargs if absent in crossovers, mutations, coeff_opts
        for variation_list in [crossovers, mutations, coeff_opts]:
            for i in range(len(variation_list)):
                if "kwargs" not in variation_list[i]:
                    variation_list[i]["kwargs"] = dict()

        # same for selection
        if "kwargs" not in selection:
            selection["kwargs"] = dict()

        # initialize some state variables
        self.population = list()

        self.archive = []

        self.num_evals = 0
        self.start_time, self.elapsed_time = 0, 0
        self.best_of_gens = list()
        self.avg_of_gens = list()
        self.memory = None

    def _must_terminate(self) -> bool:
        """
        Determines whether a termination criterion has been reached

        Returns
        -------
        bool
          True if a termination criterion is met, else False
        """
        self.elapsed_time = time.time() - self.start_time
        if self.max_time and self.elapsed_time >= self.max_time:
            return True
        elif self.max_evals and self.num_evals >= self.max_evals:
            return True
        elif self.max_gens and self.num_gens >= self.max_gens:
            return True
        return False

    def _initialize_population(self, is_diverse_population=False):
        """
        Generates a random initial population and evaluates it
        """
        if is_diverse_population:
            population = self.gen_diverse_population()
            self.population = population
        else:
            # initialize the population
            self.population = Parallel(n_jobs=self.n_jobs)(
                delayed(generate_random_multitree)(
                    self.n_trees,
                    self.internal_nodes,
                    self.leaf_nodes,
                    max_depth=self.init_max_depth,
                )
                for _ in range(self.pop_size)
            )

        # evaluate the trees and store their fitness
        fitnesses = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fitness_function)(t) for t in self.population
        )
        fitnesses = list(map(list, zip(*fitnesses)))
        memories = fitnesses[1]
        memory = memories[0]
        for m in range(1, len(memories)):
            memory += memories[m]

        self.memory = memory

        fitnesses = fitnesses[0]

        for i in range(self.pop_size):
            self.population[i].fitness = fitnesses[i]
        # store eval cost
        self.num_evals += self.pop_size
        # store best at initialization
        best = self.population[np.argmax([t.fitness for t in self.population])]
        avg = np.mean([t.fitness for t in self.population])
        self.best_of_gens.append(deepcopy(best))
        self.avg_of_gens.append(avg)

    @cache
    def gen_diverse_population(self):
        repr_set = []
        population = []
        cnt = 0
        while len(repr_set) < self.pop_size:
            while True:
                indiv = generate_random_multitree(
                    self.n_trees,
                    self.internal_nodes,
                    self.leaf_nodes,
                    max_depth=self.init_max_depth,
                )
                indiv_sp = [
                    convert_to_sympy_round(e) for e in indiv.get_readable_repr()
                ]
                if not any(compare_multitrees(indiv_sp, other) for other in repr_set):
                    if self.verbose:
                        cnt += 1
                        print(f"Init: {cnt}/{self.pop_size} ")
                    break
            population.append(indiv)
            repr_set.append(indiv_sp)
        return population

    def calculate_diversities_for_archive(self, candidates):
        """
        Keeps structurally different individuals in the archive.
        """
        n_cand = len(candidates)
        n_arch = len(self.archive)

        other = candidates
        if len(self.archive) != 0:
            other = self.archive

        n_arch = len(other)

        cand_repr = [c.get_readable_repr() for c in candidates]
        arch_repr = [a.get_readable_repr() for a in other]

        def avg_similarity_to_archive(i):
            sims = [
                compare_multitrees_old(cand_repr[i], arch_repr[j])
                for j in range(n_arch)
            ]
            return sum(sims) / n_arch if n_arch > 0 else 0

        diversities = Parallel(n_jobs=self.n_jobs)(
            delayed(avg_similarity_to_archive)(i) for i in range(n_cand)
        )

        return diversities

    def update_archive(self, candidates, archive_flag="none"):
        """
        Updates the archive. Options for flags: none, basic, diversity.
        None means that we are in a baseline variation with no flag.
        Basic refers to the state-of-the-art archive.
        Diversity archive includes both diversity adn fitness metrics.
        """
        if archive_flag == "none":
            return

        combined = self.archive + candidates

        unique = {}
        for c in combined:
            key = tuple(c.get_readable_repr())
            if key not in unique or c.fitness > unique[key].fitness:
                unique[key] = c

        combined = list(unique.values())

        if archive_flag == "basic":
            # Keep top-N based on fitness
            combined.sort(key=lambda x: x.fitness, reverse=True)
            self.archive = combined[: self.pop_size]

        elif archive_flag == "diversity":
            candidates = combined
            fitnesses = [ind.fitness for ind in candidates]
            diversities = self.calculate_diversities_for_archive(candidates)
            diversities = [diversity * -1.0 for diversity in diversities]
            f_max, f_min = max(fitnesses), min(fitnesses)
            d_max, d_min = max(diversities), min(diversities)
            norm_fitnesses = [(f - f_min) / (f_max - f_min + 1e-8) for f in fitnesses]
            norm_diversities = [
                (d - d_min) / (d_max - d_min + 1e-8) for d in diversities
            ]
            scores = [
                0.7 * f + 0.3 * d for f, d in zip(norm_fitnesses, norm_diversities)
            ]

            sorted_inds = [
                x
                for _, x in sorted(
                    zip(scores, candidates), key=lambda x: x[0], reverse=True
                )
            ]
            self.archive = sorted_inds[: self.pop_size]

    def _perform_generation(self, is_multiobjective=False, archive_flag="none"):
        """
        Performs one generation, which consists of parent selection, offspring generation, and fitness evaluation
        """
        # select promising parents when single objective (when only considering fitness)
        if not is_multiobjective:
            sel_fun = self.selection["fun"]
            parents = sel_fun(
                self.population, self.pop_size, **self.selection["kwargs"]
            )  # Tournament selection

        # otherwise, perform multiobjective selection for parents
        if is_multiobjective:
            # evaluate diversity for current population
            fitnesses = [ind.fitness for ind in self.population]
            diversities_reverted = self.calculate_diversities(self.population)
            # in the current computation of diversity, higher values means higher similarity (naming is unconventional :/)
            # hence, to convert it into a maximization problem, we revert the sign
            diversities = [diversity * -1.0 for diversity in diversities_reverted]

            individuals = [
                Individual(
                    objectives=[fitnesses[i], diversities[i]],
                    reference=self.population[i],
                )
                for i in range(self.pop_size)
            ]

            fronts = _fast_non_dominated_sorting(individuals)

            selected = []
            for front in fronts:
                if len(selected) + len(front) <= self.pop_size:
                    selected.extend(front)
                else:
                    needed = self.pop_size - len(selected)
                    selected.extend(front[:needed])
                    break
            parents = [ind.reference for ind in selected]

        # generate offspring
        offspring_population = Parallel(n_jobs=self.n_jobs)(
            delayed(generate_offspring)(
                t,
                self.crossovers,
                self.mutations,
                self.coeff_opts,
                parents,
                self.internal_nodes,
                self.leaf_nodes,
                constraints={"max_tree_size": self.max_tree_size},
            )
            for t in parents
        )

        # evaluate each offspring and store its fitness
        fitnesses = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fitness_function)(t) for t in offspring_population
        )
        fitnesses = list(map(list, zip(*fitnesses)))
        memories = fitnesses[1]
        memory = memories[0]
        for m in range(1, len(memories)):
            memory += memories[m]
        self.memory = memory + self.memory
        fitnesses = fitnesses[0]

        # evaluate diversity of the offspring under multi-objective optimization scenario
        if is_multiobjective:
            diversities = self.calculate_diversities(offspring_population)
            if self.verbose:
                print("\nSimilarity: ", diversities)

        for i in range(self.pop_size):
            offspring_population[i].fitness = fitnesses[i]

        # Calculate archive:
        self.update_archive(offspring_population, archive_flag)

        if not archive_flag == "none":
            num_elites = max(1, int(len(self.archive) * 0.1))
            sorted_archive = sorted(self.archive, key=lambda x: x.fitness, reverse=True)
            # print("Top archive fitnesses:", [c.fitness for c in sorted_archive[:5]])
            elites = deepcopy(sorted_archive[:num_elites])
            offspring_population.sort(key=lambda x: x.fitness)
            offspring_population[:num_elites] = elites

        # store cost
        self.num_evals += self.pop_size
        # update the population for the next iteration
        self.population = offspring_population
        # update info
        self.num_gens += 1
        best = self.population[np.argmax([t.fitness for t in self.population])]
        avg = np.mean([t.fitness for t in self.population])
        self.best_of_gens.append(deepcopy(best))
        self.avg_of_gens.append(avg)

    def evolve(
        self, is_multiobjective=False, is_diverse_population=False, archive_flag="none"
    ):
        """
        Runs the evolution until a termination criterion is met;
        first, a random population is initialized, second the generational loop is started:
        every generation, promising parents are selected, offspring are generated from those parents,
        and the offspring population is used to form the population for the next generation
        """
        best_fitnesses_across_gens = []
        time_elapsed = []
        num_evals = []
        average_fitness = []

        # set the start time
        self.start_time = time.time()
        self._initialize_population(is_diverse_population=is_diverse_population)
        self.archive = []
        self.update_archive(self.population, archive_flag)
        best_fitnesses_across_gens.append(self.best_of_gens[-1].fitness)
        average_fitness.append(np.average([t.fitness for t in self.population]))
        # generational loop
        while not self._must_terminate():
            # perform one generation
            self._perform_generation(is_multiobjective, archive_flag)
            # log info
            best_fitnesses_across_gens.append(self.best_of_gens[-1].fitness)
            average_fitness.append(np.average([t.fitness for t in self.population]))
            time_elapsed.append(self.elapsed_time)
            num_evals.append(self.num_evals)

            if self.verbose:
                print(
                    "gen: {}, best of gen fitness: {:.3f},\tbest of gen size: {}".format(
                        self.num_gens,
                        self.best_of_gens[-1].fitness,
                        len(self.best_of_gens[-1]),
                    )
                )

        # Get the fitnesses of the individuals in the final population
        # to draw the Pareto front
        final_population_fitnesses = [ind.fitness for ind in self.population]

        return (
            best_fitnesses_across_gens,
            average_fitness,
            time_elapsed,
            num_evals,
            final_population_fitnesses,
        )

    def calculate_diversities(self, offspring_population):

        n = len(offspring_population)
        diversities = [0] * n
        readable_reprs = [t.get_readable_repr() for t in offspring_population]

        # Helper to compare a single pair
        def pairwise(i, j):
            sim = get_diversity_of_multitrees(readable_reprs[i], readable_reprs[j])
            return (i, j, sim)

        # Generate all unique pairs (i, j) with i < j
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Parallel computation of similarities
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(pairwise)(i, j) for i, j in pairs
        )

        # Aggregate results
        for i, j, sim in results:
            diversities[i] += sim
            diversities[j] += sim

        return diversities


def _max_pareto_dominates(individual_1: Individual, individual_2: Individual) -> bool:
    is_dominates = True
    is_strictly_better = False
    objective_num = len(individual_1.objectives)

    for i in range(objective_num):
        # Individual 1 is strictly better (greater objective value)
        if individual_1.objectives[i] > individual_2.objectives[i]:
            is_strictly_better = True

        # Individual 1 is worse in at least one objective (smaller value)
        elif individual_1.objectives[i] < individual_2.objectives[i]:
            is_dominates = False
            break

    return is_dominates and is_strictly_better


def _fast_non_dominated_sorting(population: List[Individual]):
    """
    Fast non-dominating sorting algorithm
    It sorts population into non dominated fronts and assigns the correct ranks for each individual
    #####
    Input: population (list of Individual class instances)
    Output: list of lists of ranked Individuals (variable "fronts")
    fronts[0] is the best front, fronts[-1] is the worst one
    fronts[i] is a list of ranked Individual instances
    #####
    """
    current_front = []
    current_front_individuals = []
    dominates_list = [[] for _ in range(len(population))]
    domination_counts = np.zeros(len(population))

    for i, individual_1 in enumerate(population):
        dominates = []
        domination_count = 0
        for j, individual_2 in enumerate(population):
            if i == j:
                continue
            if _max_pareto_dominates(individual_1, individual_2):
                dominates.append(j)  # Store index of the dominated individual
            elif _max_pareto_dominates(individual_2, individual_1):
                domination_count += 1

        if domination_count == 0:
            individual_1.rank = 0
            current_front.append(
                i
            )  # Again, store index of the individual in the first front
            current_front_individuals.append(individual_1)

        dominates_list[i] = dominates
        domination_counts[i] = domination_count

    # Append the first front to the fronts list
    fronts = []
    fronts.append(current_front_individuals)

    current_rank = 0
    while fronts[current_rank]:
        next_front = []
        next_front_individuals = []
        for i in current_front:
            for j in dominates_list[i]:
                individual = population[i]
                next_individual = population[j]
                domination_counts[j] -= 1  # Reduce the domination count by 1
                if domination_counts[j] == 0:
                    # Assign the individual to the next front
                    next_individual.rank = current_rank + 1
                    next_front.append(j)
                    next_front_individuals.append(next_individual)
        current_rank += 1
        current_front = next_front
        fronts.append(next_front_individuals)

    return fronts
