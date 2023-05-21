# Mohamed
import numpy as np

def distance(city1, city2):
    # Calculate the Euclidean distance between two cities
    return np.linalg.norm(city1 - city2)

def evaluate_fitness(solution, cities):
    # Calculate the total distance of the TSP tour
    num_cities = len(cities)
    total_distance = 0.0
    for i in range(num_cities - 1):
        total_distance += distance(cities[solution[i]], cities[solution[i+1]])
    total_distance += distance(cities[solution[-1]], cities[solution[0]])

    # Return the inverse of the total distance as the fitness value
    fitness = 1.0 / total_distance

    return fitness
    
def local_search(solution, cities):
    # Perform 2-opt local search to improve the TSP solution
    num_cities = len(solution)
    best_solution = solution.copy()
    best_distance = evaluate_fitness(best_solution, cities)
    improved = True
    while improved:
        improved = False
        for i in range(1, num_cities - 1):
            for j in range(i + 1, num_cities):
                new_solution = best_solution.copy()
                new_solution[i:j] = list(reversed(best_solution[i:j]))
                new_distance = evaluate_fitness(new_solution, cities)
                if new_distance < best_distance:
                    best_solution = new_solution
                    best_distance = new_distance
                    improved = True
    return best_solution


def bat_algorithm(num_bats, num_iterations, cities, min_frequency, max_frequency, min_loudness, max_loudness, alpha, gamma):

    num_cities = len(cities)
    dimension = num_cities

    # Initialize bats
    bats = np.zeros((num_bats, dimension), dtype=int)
    velocities = np.zeros((num_bats, dimension))
    frequencies = np.random.uniform(min_frequency, max_frequency, num_bats)
    loudness = np.random.uniform(min_loudness, max_loudness, num_bats)

    # Initialize the best solution
    best_solution = np.random.permutation(num_cities)
    best_fitness = evaluate_fitness(best_solution, cities)

    # Main loop
    for t in range(num_iterations):
        for i in range(num_bats):
            # Update bat's velocity and frequency
            velocities[i] += (bats[i] - best_solution) * frequencies[i]
            frequencies[i] = min_frequency + (max_frequency - min_frequency) * np.random.random()

            # Update bat's position
            bats[i] = np.clip(bats[i] + velocities[i], 0, num_cities-1)

            # Perform random walk exploration
            if np.random.random() > loudness[i]:
                bats[i] = np.random.permutation(num_cities)

            # Apply local search
            if np.random.random() < gamma:
                bats[i] = local_search(bats[i], cities)

            # Evaluate new solution
            fitness = evaluate_fitness(bats[i], cities)

            # Update the best solution
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = bats[i]

        print(f"Iteration {t+1}: Best fitness = {best_fitness}")

    return best_solution, best_fitness

# Example usage
num_bats = 8
num_iterations = 100
min_frequency = 0.3
max_frequency = 0.7
min_loudness = 0.0
max_loudness = 1.0
alpha = 0.5
gamma = 0.5

# Generate a random set of cities
num_cities = 20
#cities = np.random.rand(num_cities, 3)  # 2D cities with random coordinates
cities = np.random.uniform(0, 100, (num_cities, 2))

best_solution, best_fitness = bat_algorithm(num_bats, num_iterations, cities, min_frequency, max_frequency, min_loudness, max_loudness, alpha, gamma)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
