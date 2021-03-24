from typing import List

Vector = List[float]

height_weight_age = [70,
                     170,
                     40]

grades = [95,
          80,
          75,
          62]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "두 벡터의 길이가 같아야 함"
    return [v_i + w_i for v_i, w_i in zip(v, w)]


assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "두 벡터의 길이가 같아야 함"
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors: List[Vector]) -> Vector:
    return [sum(vector[i] for vector in vectors) for i in range(len(vectors[0]))]


def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "벡터의 길이가 같아야 함"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


assert dot([1, 2, 3], [4, 5, 6]) == 32


