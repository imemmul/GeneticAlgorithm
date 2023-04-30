import numpy as np
import random
class GeneticAlgorithm():
    def __init__(self, requirements, proficiency_levels, num_gen=100, population_size=50, mutation_rate=0.1) -> None:
        self.num_gen = num_gen
        self.pop_size = population_size
        self.requirements = requirements
        self.proficiency = proficiency_levels
        self.num_tasks = self.requirements.shape[1]
        self.num_students = self.proficiency.shape[1]
        self.mutation_rate = mutation_rate
    
    def run(self):
        # requirements.shape[0] = number of tasks
        # requirements.shape[1] = number of skillsets
        # proficiency.shape[0] = number of students
        # proficiency.shape[1] = number of skillsets
        # print(self.requirements)
        population = self.initialize_population()
        for i in range(self.num_gen):
            new_population = []
            for j in range(self.pop_size):
                parents = self.select_parents(population)
                child = self.crossover(parents)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
        best_solution = max(population, key=self.fitness_function)
        best_fitness = self.fitness_function(best_solution)
        return best_solution, best_fitness

    def fitness_function(self, solution):
        total_fitness = 0
        for p in range(self.num_tasks):
            project_fitness = 0
            assigned_students = []
            for s in range(self.num_students):
                if solution[s][p] == 1:
                    assigned_students.append(s)
                    project_fitness += sum(self.requirements[p] * self.proficiency[s])
            if len(assigned_students) == 0:
                return 0
            total_fitness += project_fitness / len(assigned_students)
        return total_fitness
    
    def initialize_population(self):
        population = []
        for i in range(self.pop_size):
            solution = np.zeros((self.num_students, self.num_tasks))
            for p in range(self.num_tasks):
                num_assigned = random.randint(1, 3)
                assigned_students = random.sample(range(self.num_students), num_assigned)
                for s in assigned_students:
                    solution[s][p] = 1
            population.append(solution)
        # print(population)
        return np.array(population)
    def select_parents(self, population):
        fitness_values = [self.fitness_function(solution) for solution in population]
        sum_fitness = sum(fitness_values)
        probabilities = [fitness / sum_fitness for fitness in fitness_values]
        flat_population = population.reshape((population.shape[0], -1))

        # Select indices from the flattened population using the probabilities
        indices = np.random.choice(range(len(flat_population)), size=2, replace=False, p=probabilities)

        # Convert the indices back to 3D indices
        indices_3d = np.unravel_index(indices, population.shape[:2])

        # Select the individuals from the population using the 3D indices
        parents = population[indices_3d]
        return parents
    def crossover(self, parents):
        print(f"parents {parents}")
        parent1, parent2 = parents
        print(parent1)
        child = np.zeros((self.num_students, self.num_tasks))
        for p in range(self.num_tasks):
            for s in range(self.num_students):
                if parent1[s][p] == 1 or parent2[s][p] == 1:
                    child[s][p] = 1
        return child
    def mutate(self, solution):
        for p in range(self.num_tasks):
            for s in range(self.num_students):
                if solution[s][p] == 1 and random.random() < self.mutation_rate:
                    solution[s][p] = 0
                elif solution[s][p] == 0 and random.random() < self.mutation_rate:
                    solution[s][p] = 1
        return solution