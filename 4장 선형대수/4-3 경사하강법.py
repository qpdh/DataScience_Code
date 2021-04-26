from typing import Callable
from typing import List

Vector = List[float]


def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f: Callable[[Vector], float], v: Vector, i: int, h: float) -> float:
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


import random
from scratch.linear_algebra import distance, add, scalar_multiply


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]


v = [random.uniform(-10, 10) for i in range(3)]
#
# for epoch in range(1000):
#     grad = sum_of_squares_gradient(v)
#     v = gradient_step(v, grad, -0.01)
#     print(epoch, v)
#
# assert distance(v, [0, 0, 0]) < 0.001

for epoch in range(1000):
    grad = estimate_gradient(lambda f: (f[0] - 1) ** 2 + (f[1] - 2) ** 2 + (f[2] - 3) ** 2, v)
    v = gradient_step(v, grad, -0.01)
    print(epoch, v)

assert distance(v, [1, 2, 3]) < 0.001
import math

v = [random.uniform(-10, 10) for i in range(1)]
for epoch in range(1000):
    grad = estimate_gradient(lambda f: (math.e ** f[0] + (f[0] - 2) ** 2), v)
    v = gradient_step(v, grad, -0.01)
    print(epoch, v)

print(v)


