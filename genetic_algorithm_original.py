import numpy as np
import random
class GeneticAlgorithm():
    def __init__(self, requirements, proficiency_levels, num_gen=50, population_size=50, mutation_rate=0.1) -> None:
        self.num_gen = num_gen # setting num_gen default 50
        self.pop_size = population_size # default 50
        self.requirements = requirements
        self.proficiency = proficiency_levels
        self.num_tasks = self.requirements.shape[0] 
        self.num_students = self.proficiency.shape[1]
        self.mutation_rate = mutation_rate #default 20
        self.is_valid = True
        
    def run(self):
        population = self.initialize_population() # init pop
        for generation in range(self.num_gen): # iterate through num_gen
            fitness_values = [self.calculate_fitness(chromosome) for chromosome in population] # calculate fitness values for every chromosomes
            
            while self.is_valid:
                if all(value == 0 for value in fitness_values):
                    population = self.initialize_population()
                    fitness_values = [self.calculate_fitness(chromosome) for chromosome in population]
                else:
                    self.is_valid = False 
                    break # calculate fitness values for every chromosomes
            best_fitness = max(fitness_values) # get the best fitness and best chromosome 
            best_chromosome = population[fitness_values.index(best_fitness)]
            print(f"Generation {generation + 1}: Fitness = {best_fitness}")

            new_population = [best_chromosome]  # here comes the elitism I should keep best fitness valued chromosome unchanged
            # apply selection crossover and mutation these function mostly inspired from Kie Codes Genetic Algorithms Coding video.
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.selection(population, fitness_values)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1, self.mutation_rate)
                child2 = self.mutation(child2, self.mutation_rate)
                new_population.extend([child1, child2])

            population = new_population
        return best_chromosome

    def calculate_fitness(self, chromosome):
        total_proficiency = 0
        for project, assigned in enumerate(chromosome):
            if assigned:
                total_proficiency += np.sum(self.requirements[project] * self.proficiency) # adding proficiencies to total_prof
        # TODO It is mandatory to assign at least one student to a project, while a maximum of three students can be assigned to any given project. In addition to this, one student can be assigned to at most two projects.
        # TODO Does below code applies constraints ?
        skills = np.dot(chromosome, self.requirements) # to use in constraint of 
        # print(f"chromosome {skills}")
        tasks = np.sum(chromosome)

        if np.any(skills > 3) or tasks > 2: # does this if apply given constraints ?
            # print("constrained asserted")
            return 0
        else:
            # print("proficiency passed")
            return total_proficiency
    
    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            chromosome = np.zeros(self.num_tasks, dtype=int) # initiliazing chromosomes (solutions) with zeros
            num_assigned_projects = random.randint(1, self.num_tasks) # randomized num of assigned projects
            assigned_projects = random.sample(range(self.num_tasks), num_assigned_projects) # take random sample from num_assigned_projects
            for project in assigned_projects:
                chromosome[project] = 1 # assign projects
            population.append(chromosome) # creating array of chromosomes population
        return population 
        
    
    def selection(self, population, fitness_values):
        # selects two individuals from the population based on their fitness values
        #  implements a form of selection mechanism that favors individuals with higher fitness values, allowing them to have a higher chance of being chosen for reproduction.
        
        # weights = fitnss_values does it
        if all(value == 0 for value in fitness_values):
            selected_indices = random.sample(range(len(population)), k=2)
        else:
            selected_indices = random.choices(range(len(population)), weights=fitness_values, k=2)
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1) # selecting random point that where to makes crossover from
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    
    def mutation(self, chromosome, mutation_rate):
        # basic mutation implementation
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # Flip the bit
        return chromosome