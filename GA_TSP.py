import random
import time

######### Input ############
n = int(input())
e = [0]
l = [0]
d = [0]

for i in range(n):
    _ = list(map(int, input().split()))
    e.append(_[0])
    l.append(_[1])
    d.append(_[2])

time_matrix = []
for i in range(n + 1):
    _ = list(map(int, input().split()))
    time_matrix.append(_)

start_time = time.process_time()

parents = []
best_fitness = float('inf')
best_tour = []

def check_feasible(real_time, p):
    try:
        real_time = 0
        for i in range(n):
            if i == 0:
                real_time += d[0] + time_matrix[0][p[i]]
                real_time = max(real_time, e[p[i]])
            else:
                real_time += d[p[i-1]] + time_matrix[p[i-1]][p[i]]
                real_time = max(real_time, e[p[i]])
            if real_time > l[p[i]]:
                return False
        return True
    except Exception:
        return False

def greedy_solution():
    greedy_path = sorted(range(1, n + 1), key=lambda x: e[x])
    if check_feasible(0, greedy_path):
        return greedy_path
    return None

def create_shuffle():
    lst = [i for i in range(1, n + 1)]
    random.shuffle(lst)
    return lst

def some_initial_population():
    population = []
    for _ in range(100):
        candidate = create_shuffle()
        if check_feasible(0, candidate):
            population.append(candidate)
        if len(population) >= 10:
            break
    greedy = greedy_solution()
    if greedy and greedy not in population:
        population.append(greedy)
    return population

def fitness(tour):
    total_time = time_matrix[0][tour[0]]
    for i in range(1, n):
        total_time += time_matrix[tour[i-1]][tour[i]] + d[tour[i-1]]
    return total_time

def evaluate(population):
    fitnesses = [fitness(tour) for tour in population]
    min_fit = min(fitnesses)
    best_idx = fitnesses.index(min_fit)
    global best_fitness, best_tour
    if min_fit < best_fitness:
        best_fitness = min_fit
        best_tour = population[best_idx]
    total_fit = sum(fitnesses)
    if total_fit == 0:  #avoid zero division
        return [1 / len(fitnesses)] * len(fitnesses)  # Assign equal probabilities if total fitness is 0
    normalized_fitness = [1 / f if f != 0 else 0 for f in fitnesses]  # Higher fitness -> Lower probability
    total_normalized = sum(normalized_fitness)
    return [f / total_normalized for f in normalized_fitness]

def select_parents(population, fitness_probs):
    return random.choices(population, weights=fitness_probs, k=len(population))

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]
    pointer = 0
    for i in range(size):
        if child[i] is None:
            while parent2[pointer] in child:
                pointer += 1
            child[i] = parent2[pointer]
    return child

def mutate(tour, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def genetic_algorithm(num_generations, population_size, crossover_rate, mutation_rate):
    population = some_initial_population()
    for _ in range(num_generations):
        fitness_probs = evaluate(population)
        parents = select_parents(population, fitness_probs)
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1[:]
            child = mutate(child, mutation_rate)
            if check_feasible(0, child):
                next_generation.append(child)
        population = next_generation if next_generation else population
    evaluate(population)




if __name__ == '__main__':
    start_time
    end_time = start_time + 180
    genetic_algorithm(num_generations=100, population_size=20, crossover_rate=0.8, mutation_rate=0.1)
    print(n)
    print(*best_tour)



