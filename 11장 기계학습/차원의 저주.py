import random
from typing import List
from scratch.linear_algebra import *
from scratch.statistics import *

Vector = List[float]


def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]


def random_distances(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]


dimensions = range(1, 101, 5)
avg_distance = []
min_distance = []

random.seed(0)
for dim in dimensions:
    distances = random_distances(dim, 10000)
    avg_distance.append(mean(distances))
    min_distance.append(min(distances))
    print(dim, min(distances), mean(distance))
