import pickle
import numpy as np
import os
from genetic_algorithm import GeneticAlgorithm
DATASET_DIR = "datasets/"
DATASET_NAME = "dataset1/"

with open(os.path.join(DATASET_DIR, DATASET_NAME, "requirements.pkl"), "rb") as f:
    requirements: np.ndarray = pickle.load(f)

with open(os.path.join(DATASET_DIR, DATASET_NAME, "proficiency_levels.pkl"), "rb") as f:
    proficiency_levels: np.ndarray = pickle.load(f)

ga = GeneticAlgorithm(num_gen=100, population_size=50, requirements=requirements, proficiency_levels=proficiency_levels)
# student_skills = np.random.randint(0, 101, (20, 5))
ga.run()
# print(requirements.shape)
# print(requirements)

# print("-" * 50)

# print(proficiency_levels.shape)
# print(proficiency_levels)
