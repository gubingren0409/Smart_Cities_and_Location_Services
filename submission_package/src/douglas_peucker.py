from __future__ import division
from math import sqrt, pow
import matplotlib.pyplot as plt
#import util as util
from utils import *
#THRESHOLD = 5  # 阈值5m


# 计算点a到点b c所在直线的距离
def point2LineDistance(point_a, point_b, point_c):
    # 首先计算b c 所在直线的斜率和截距
    if point_b[0] == point_c[0]:
        return 9999999
    slope = (point_b[1] - point_c[1]) / (point_b[0] - point_c[0])
    intercept = point_b[1] - slope * point_b[0]
    # 计算点a到b c所在直线的距离
    distance = abs(slope * point_a[0] - point_a[1] + intercept) / sqrt(1 + pow(slope, 2))
    return distance


class DouglasPeuker(object):
    def __init__(self, THRESHOLD):
        self.threshold = THRESHOLD
        self.qualify_list = list()
        self.disqualify_list = list()

    # 抽稀
    def diluting(self, point_list):
        if len(point_list) < 3:
            self.qualify_list.extend(point_list[::-1])
        else:
            # 找到与收尾两点连线距离最大的点
            max_distance_index, max_distance = 0, 0
            for index, point in enumerate(point_list):
                if index in [0, len(point_list) - 1]:
                    continue
                distance = point2LineDistance(point, point_list[0], point_list[-1])
                if distance > max_distance:
                    max_distance_index = index
                    max_distance = distance

            # 若最大距离小于阈值，则去掉所有中间点。 反之，则将曲线按最大距离点分割
            if max_distance < self.threshold:
                self.qualify_list.append(point_list[-1])
                self.qualify_list.append(point_list[0])
            else:
                # 将曲线按最大距离的点分割成两段
                sequence_a = point_list[:max_distance_index]
                sequence_b = point_list[max_distance_index:]

                for sequence in [sequence_a, sequence_b]:
                    if len(sequence) < 3 and sequence == sequence_b:
                        self.qualify_list.extend(sequence[::-1])
                    else:
                        self.disqualify_list.append(sequence)

    def reduction(self, point_list):
        self.diluting(point_list)
        while len(self.disqualify_list) > 0:
            self.diluting(self.disqualify_list.pop())
        # print(len(point_list))
        # print(self.qualify_list)
        # print("after",len(self.qualify_list))
        # x = [i[0] for i in point_list]
        # y = [i[1] for i in point_list]
        # plt.plot(x, y)
        # resx = [i[0] for i in self.qualify_list]
        # resy = [i[1] for i in self.qualify_list]
        # plt.plot(resx, resy, linestyle=':')
        # plt.show()
