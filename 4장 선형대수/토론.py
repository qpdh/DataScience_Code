# 가변 갯수 함수 파라미터 처리

"""
임의의 신체 치수를 여러번 측정하여 평균낸 값을 받는 함수 getbody 함수를 만들어
이름, BMI, 첫번째 측정장소를  함께 출력하시오.
파라미터는 다음 조건을 만족한다.
1) name은 첫 파라미터로 필수로 주어진다.
2) 평균값을 구한 신체사이즈 height, weight, waist, footsize  등의 값은 임의의 순서로 파라미터를 받을 수 있다.
3) 가변갯수의 측정장소들이 올수 있다.

---- 출력문
print(getbody("Tommy", 'Jeongwangdong', 'Seoul', 'Incheon', footsize=265, weight=80, height=180))
"""


def getbody(name, *spaces, **sizes):
    return name, spaces[0], round(sizes['weight'] / ((sizes['height'] / 100) ** 2), 2)


print(getbody("Tommy", 'Jeongwangdong', 'Seoul', 'Incheon', footsize=265, weight=80, height=180))

"""
아래와 같이 만들어진 l을 변형하여, 다시 a와 b형태로 a2, b2를 만들어 출력하시오. 
a = "abcde"
b = [1,3,5,6,9]
l = list(zip(a,b))
print(a2, b2)
"""
a = "abcde"
b = [1, 3, 5, 6, 9]
l = list(zip(a, b))
a2, b2 = zip(*l)
a2 = list(a2)
print(a2, b2)

"""
# 실습 4.1 –(2)
def vector_sum(vectors: List[Vector]) -> Vector:

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
"""
from typing import List

Vector = List[float]


def vector_sum(vectors: List[Vector]) -> Vector:
    return [sum(vector[i] for vector in vectors) for i in range(len(vectors[0]))]


assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

"""
# 실습 4.2 – (1) matrix shape

"""
Matrix = List[Vector]

from typing import Tuple


def shape(A: Matrix) -> Tuple[int, int]:
    return (len(A), len(A[0]))


# fill in with your code
assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 rows, 3 columns

# 실습 4.2. (2) 함수를 이용해 행렬만들기
# (i,j)원소의 값을 entry_fn(i, j)로 설정하는 i x j 행렬을 리턴하는 함수를 완성하시오.
from typing import Callable


def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(i, j) for i in range(num_cols)] for j in range(num_rows)]


print(make_matrix(3, 5, lambda i, j: (i,j,)))
