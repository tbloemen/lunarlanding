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

from genepro.node import Node
from genepro.variation import *
from genepro.selection import tournament_selection
from compare_expressions import compare_multitrees

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
  def __init__(self,
    # required settings
    fitness_function : Callable[[Node], float],
    internal_nodes : list,
    leaf_nodes : list,
    # optional evolution settings
    n_trees : int,
    pop_size : int=256,
    init_max_depth : int=4,
    max_tree_size : int=64,
    crossovers : list=[{"fun":subtree_crossover, "rate": 0.5}],
    mutations : list= [{"fun":subtree_mutation, "rate": 0.5}],
    coeff_opts : list = [{"fun":coeff_mutation, "rate": 0.5}],
    selection : dict={"fun":tournament_selection,"kwargs":{"tournament_size":8}},
    # termination criteria
    max_evals : int=None,
    max_gens : int=100,
    max_time : int=None,
    # other
    n_jobs : int=4,
    verbose : bool=False,
    ):

    # set parameters as attributes
    _, _, _, values = inspect.getargvalues(inspect.currentframe())
    values.pop('self')
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
    self.num_gens = 0
    self.num_evals = 0
    self.start_time, self.elapsed_time = 0, 0
    self.best_of_gens = list()

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

  def _initialize_population(self):
    """
    Generates a random initial population and evaluates it
    """
    # initialize the population
    self.population = Parallel(n_jobs=self.n_jobs)(
        delayed(generate_random_multitree)(self.n_trees, 
          self.internal_nodes, self.leaf_nodes, max_depth=self.init_max_depth )
        for _ in range(self.pop_size))

    for count, individual in enumerate(self.population):
      individual.get_readable_repr()

    # evaluate the trees and store their fitness
    fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in self.population)
    fitnesses = list(map(list, zip(*fitnesses)))
    memories = fitnesses[1]
    memory = memories[0]
    for m in range(1,len(memories)):
      memory += memories[m]

    self.memory = memory

    fitnesses = fitnesses[0]

    for i in range(self.pop_size):
      self.population[i].fitness = fitnesses[i]
    # store eval cost
    self.num_evals += self.pop_size
    # store best at initialization
    best = self.population[np.argmax([t.fitness for t in self.population])]
    self.best_of_gens.append(deepcopy(best))

  def _perform_generation(self, is_multiobjective = False):
    """
    Performs one generation, which consists of parent selection, offspring generation, and fitness evaluation
    """
    # evaluate diversity for current population
    fitnesses = [ind.fitness for ind in self.population]
    diversities_reverted = self.calculate_diversities(self.population)
    diversities = [diversity * -1.0 for diversity in diversities_reverted]

    # select promising parents when single objective (aka fitness)
    if not is_multiobjective:
      sel_fun = self.selection["fun"]
      parents = sel_fun(self.population, self.pop_size, **self.selection["kwargs"])

    # otherwise, perform multiobjective selection for parents
    if is_multiobjective:
      individuals = [
        Individual(objectives=[fitnesses[i], diversities[i]], reference=self.population[i])
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
    offspring_population = Parallel(n_jobs=self.n_jobs)(delayed(generate_offspring)
      (t, self.crossovers, self.mutations, self.coeff_opts, 
      parents, self.internal_nodes, self.leaf_nodes,
      constraints={"max_tree_size": self.max_tree_size}) 
      for t in parents)

    # evaluate each offspring and store its fitness 
    fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in offspring_population)
    fitnesses = list(map(list, zip(*fitnesses)))
    memories = fitnesses[1]
    memory = memories[0]
    for m in range(1,len(memories)):
      memory += memories[m]
    self.memory = memory + self.memory
    fitnesses = fitnesses[0]

    # evaluate diversity
    diversities = self.calculate_diversities(offspring_population)
    if self.verbose:
      print("\nDIVERSION: ", diversities)

    for i in range(self.pop_size):
      offspring_population[i].fitness = fitnesses[i]
    
    # store cost
    self.num_evals += self.pop_size
    # update the population for the next iteration
    self.population = offspring_population
    # update info
    self.num_gens += 1
    best = self.population[np.argmax([t.fitness for t in self.population])]
    self.best_of_gens.append(deepcopy(best))


  def evolve(self, is_multiobjective = False):
    """
    Runs the evolution until a termination criterion is met;
    first, a random population is initialized, second the generational loop is started:
    every generation, promising parents are selected, offspring are generated from those parents, 
    and the offspring population is used to form the population for the next generation
    """
    best_fitnesses_across_gens = []

    # set the start time
    self.start_time = time.time()

    self._initialize_population()

    # generational loop
    while not self._must_terminate():
      # perform one generation
      self._perform_generation(is_multiobjective)
      # log info
      if self.verbose:
        best_fitnesses_across_gens.append(self.best_of_gens[-1].fitness)
        print("gen: {},\tbest of gen fitness: {:.3f},\tbest of gen size: {}".format(
            self.num_gens, self.best_of_gens[-1].fitness, len(self.best_of_gens[-1]))
            )
    
    return best_fitnesses_across_gens
  

  def calculate_diversities(self, offspring_population):
    
    n = len(offspring_population)
    diversities = [0] * n
    readable_reprs = [t.get_readable_repr() for t in offspring_population]

    # Helper to compare a single pair
    def pairwise(i, j):
        sim = compare_multitrees(readable_reprs[i], readable_reprs[j])
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


def _max_pareto_dominates( individual_1: Individual, individual_2: Individual) -> bool:
    is_dominates = True
    is_strictly_better = False
    objective_num = len(individual_1.objectives)

    for i in range(objective_num):
        # Individual 1 is strictly better (greater objective value)
        if individual_1.objectives[i] > individual_2.objectives[i]: 
            is_strictly_better = True

        # Individual 1 is worse in at least one objective (smaller value)
        elif individual_1.objectives[i] < individual_2.objectives[i]: 
            is_dominates = False
            break
            
    return is_dominates and is_strictly_better


def _fast_non_dominated_sorting(population: List[Individual]):
    '''
    Fast non-dominating sorting algorithm
    It sorts population into non dominated fronts and assigns the correct ranks for each individual
    #####
    Input: population (list of Individual class instances)
    Output: list of lists of ranked Individuals (variable "fronts")
    fronts[0] is the best front, fronts[-1] is the worst one
    fronts[i] is a list of ranked Individual instances
    #####
    '''
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
          dominates.append(j) # Store index of the dominated individual
        elif _max_pareto_dominates(individual_2, individual_1):
          domination_count += 1

      if domination_count == 0:
        individual_1.rank = 0
        current_front.append(i) # Again, store index of the individual in the first front 
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
          domination_counts[j] -= 1 # Reduce the domination count by 1
          if domination_counts[j] == 0:
            # Assign the individual to the next front
            next_individual.rank = current_rank + 1
            next_front.append(j)
            next_front_individuals.append(next_individual)
      current_rank += 1
      current_front = next_front
      fronts.append(next_front_individuals)

    return fronts
