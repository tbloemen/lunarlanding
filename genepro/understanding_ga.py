import random

# Problem setup
items = [
    {"value": 60, "weight": 10},
    {"value": 100, "weight": 20},
    {"value": 120, "weight": 30}
]
capacity = 50
population_size = 10
generations = 50
mutation_rate = 0.1

def generate_individual():
    return [random.randint(0, 1) for _ in items]

def fitness(individual):
    total_weight = total_value = 0
    for gene, item in zip(individual, items):
        if gene == 1:
            total_weight += item["weight"]
            total_value += item["value"]
    if total_weight > capacity:
        return 0  # penalty for overweight
    return total_value

def selection(population):
    # Tournament selection
    tournament = random.sample(population, 3)
    return max(tournament, key=fitness)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

def mutate(individual):
    return [
        gene if random.random() > mutation_rate else 1 - gene
        for gene in individual
    ]

# Initialize population
population = [generate_individual() for _ in range(population_size)]

# Evolution loop
for generation in range(generations):
    new_population = []
    for _ in range(population_size):
        parent1 = selection(population)
        parent2 = selection(population)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)
    population = new_population

# Get the best solution
best = max(population, key=fitness)
print("Best individual:", best)
print("Best fitness:", fitness(best))
