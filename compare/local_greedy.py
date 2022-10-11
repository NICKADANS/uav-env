from copy import deepcopy

import numpy as np
import torch
import sys
sys.path.append('..')
from poi import PoI


# 通过全局贪婪算法选择行为，默认环境不包含障碍物
"""
    以data/pois.npy中的273个poi为例
    无人机个数为1-10的时候，PoI的采集个数为：[107, 192, 226, 260, 272, 268, 267, 267, 273, 273]
    采集效率为：[39.1%, 70.3%, 82.8%, 95.2%, 之后至少 97.8%]
"""
def select_actions(env):
    actions = []
    targets = {}
    for uav in env.uavs:
        mindistance = 2000  # 最短距离
        target = None  # 保存POI下标
        for i in range(len(env.pois)):
            distance = np.sqrt((uav.x - env.pois[i].x) ** 2 + (uav.y - env.pois[i].y) ** 2)
            if targets.get(i) is None and env.pois[i].done == 0 and distance < mindistance:
                mindistance = distance
                target = i
        # 将垓下标加入字典
        targets[target] = 1
        if target is None:
            actions.append(2)
            continue
        else:
            target = env.pois[target]
        dx = target.x - uav.x
        dy = target.y - uav.y
        # actions : [(-20, 0), (20, 0), (0, 0), (0, 20), (0, -20), (10, 10), (-10, 10), (10, -10), (-10, -10)]
        index = None
        if -10 < dy < 10 and -10 < dx < 10:
            index = 2
        elif dy > 20 or dy < -20 or dx > 20 or dx < -20:
            if dy > 20:
                index = 3
            elif dy < -20:
                index = 4
            elif dx > 20:
                index = 1
            elif dx < -20:
                index = 0
        else:
            if dx > 0 and dy > 0:
                index = 5
            elif dx < 0 and dy > 0:
                index = 6
            elif dx > 0 and dy < 0:
                index = 7
            elif dx < 0 and dy < 0:
                index = 8
        actions.append(index)
    return actions
