from collections import Counter
from typing import List

num_friends = [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 6, 6]


def mode(x: List[float]) -> List[float]:
    counter = Counter(num_friends)
    max_friends = counter.most_common(1)[0][1]
    print([n[0] for n in counter.most_common() if n[1] == max_friends])
    return [n[0] for n in counter.most_common() if n[1] == max_friends]


assert set(mode(num_friends)) == {1, 6}
