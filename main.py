import pickle
import numpy as np
import os
from genetic_algorithm_original import GeneticAlgorithm
from roulette_ga import GeneticAlgorithmRoulette
from tournament_ga import GeneticAlgorithmTournament
DATASET_DIR = "datasets/"
DATASET_NAME = "dataset3/"

with open(os.path.join(DATASET_DIR, DATASET_NAME, "requirements.pkl"), "rb") as f:
    requirements: np.ndarray = pickle.load(f)

with open(os.path.join(DATASET_DIR, DATASET_NAME, "proficiency_levels.pkl"), "rb") as f:
    proficiency_levels: np.ndarray = pickle.load(f)
print("requirements")
print(requirements)
print("-"*50)
print("proficiency_levels")
print(proficiency_levels)
print("Original Implementation")
ga = GeneticAlgorithm(num_gen=100, population_size=50, requirements=requirements, proficiency_levels=proficiency_levels)
print(ga.run())
print("Roulette Wheel Selection Implementation")
ga_roulette = GeneticAlgorithmRoulette(num_gen=100, population_size=50, requirements=requirements, proficiency_levels=proficiency_levels)
print(ga_roulette.run())
print("Tournament Selection Implementation")
ga_tournament = GeneticAlgorithmTournament(num_gen=100, population_size=50, requirements=requirements, proficiency_levels=proficiency_levels)
print(ga_tournament.run())
# print("Greedy Implementation")
# ga_greedy = GeneticAlgorithmGreedy(num_gen=100, population_size=50, requirements=requirements, proficiency_levels=proficiency_levels)
# print(ga_greedy.run())
