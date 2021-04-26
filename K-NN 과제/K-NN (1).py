from scratch.linear_algebra import distance
from collections import Counter
from statistics import mean
import math, random
import matplotlib.pyplot as plt
import csv
import pandas as pd
from typing import TypeVar, List, Tuple


# print(knn_classify(3, cities, [-80, 30]))


# def plot_cities():
#     plots = {"Java": ([], []), "Python": ([], []), "R": ([], [])}
#
#     markers = {"Java": "o", "Python": "s", "R": "^"}
#     colors = {"Java": "r", "Python": "b", "R": "g"}
#
#     for (longitude, latitude), language in cities:
#         plots[language][0].append(longitude)
#         plots[language][1].append(latitude)
#
#     for language, (x, y) in plots.items():
#         plt.scatter(x, y, color=colors[language], marker=markers[language],
#                     label=language, zorder=10)
#
#     plt.legend(loc=0)
#     plt.axis([-130, -60, 20, 55])
#     plt.title("Favorite Programming Languages")
#     plt.show()
#
#
# def clasifiy_and_plot_grid(k=1):
#     plots = {"Java": ([], []), "Python": ([], []), "R": ([], [])}
#
#     markers = {"Java": "o", "Python": "s", "R": "^"}
#     colors = {"Java": "r", "Python": "b", "R": "g"}
#
#     for longitude in range(-130, -60):
#         for latitude in range(20, 55):
#             predicted_language = knn_classify(k, cities, [longitude, latitude])
#             plots[predicted_language][0].append(longitude)
#             plots[predicted_language][1].append(latitude)
#
#     for language, (x, y) in plots.items():
#         plt.scatter(x, y, color=colors[language], marker=markers[language],
#                     label=language, zorder=0)
#
#     plt.legend(loc="upper right")
#     plt.axis([-130, -60, 20, 55])
#     plt.title(str(k) + "-Nearest Neighbor Programming Languages")
#     plt.show()


# plot_cities()
# clasifiy_and_plot_grid(3)

def rqw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner


def majority_vote(labels):
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]

    # 1등이 몇명인가?
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])


def knn_classify(k, labeled_points, new_point):
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance(point_label[0], new_point))

    k_nearest_labels = [label for _, label in by_distance[:k]]
    # print(k_nearest_labels)
    return sum(k_nearest_labels) / k


#    return majority_vote(k_nearest_labels)


# 데이터를 읽어 bmd_data에 저장
df = pd.read_csv("bmd.csv")
# kg, height, bmd, age 추출
df = df[['weight_kg', 'height_cm', 'bmd', 'age']]

bmd_data = []
# 데이터를 형식에 맞게 조정
for i in range(len(df.index)):
    bmd_data.append(([df.loc[i].weight_kg, df.loc[i].height_cm, df.loc[i].bmd], df.loc[i].age))
# print(bmd_data)
# bmd_data = [([weight_kg, height_cm, bmd, age]) for weight_kg, height_cm, bmd, age in bmd_data]

X = TypeVar('X')


# 데이터를 0.75 0.25 분할
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]


bmd_train, bmd_test = split_data(bmd_data, 0.75)
# print(len(bmd_train), len(bmd_test))

predict_y = []
for _ in range(4):
    predict_y.append(list())

for k in range(4):
    for data, y in bmd_test:
        predict_y[k].append(float(knn_classify(k * 2 + 1, bmd_train, data)))

test_y = [y for _, y in bmd_test]
sse_list = []

for i in range(4):
    sse = 0
    for j in range(len(predict_y)):
        sse += (predict_y[i][j] - test_y[j]) ** 2
    sse_list.append(sse)


min_k = sse_list.index(min(sse_list))*2+1
min_t = (min_k-1)//2
min_error = min(sse_list)
print(min_k,min_error)

#print(bmd_test)
for test_index in range(len(bmd_test)-10,len(bmd_test)):
    print(test_index, predict_y[min_t][test_index], test_y[test_index])