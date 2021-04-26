from typing import List
from collections import Counter


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def median(v: List[float]) -> float:
    v.sort()
    if len(v) % 2 == 0:
        return (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
    else:
        return v[len(v) // 2]


assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2


def qunatils(xs: List[float], p: float) -> float:
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


num_friends = [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 6, 6]


def mode(x: List[float]) -> List[float]:
    counter = Counter(num_friends)
    max_num = counter.most_common(1)[0][1]

    result = []
    for num, fre in counter.most_common():
        if fre == max_num:
            result.append(num)
    print(result)
    return result


assert set(mode(num_friends)) == {1, 6}

from scratch.linear_algebra import sum_of_squares


def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def variance(xs: List[float]) -> float:
    assert len(xs) >= 2, "둘 이상의 원소가 있어야 함"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


print(variance([1, 2, 3]))

import math


def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))


def interquartile_range(xs: List[float]) -> float:
    return quantile(xs, 0.75) - quantils(xs, 0.25)


def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "같아야 함"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


assert 22.42 < covariance(num_friends, daily_minutes) < 22.43

