from typing import Tuple
from typing import List
from typing import Callable

Matrix = List[List[float]]
Vector = List[float]


def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def get_row(A: Matrix, i: int) -> Vector:
    return A[i]


def get_col(A: Matrix, i: int) -> Vector:
    return [A_i[i] for A_i in A]


def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(i, j) for i in range(num_cols)] for j in range(num_rows)]


def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


