from copy import deepcopy

import numpy as np
import sys
sys.path.append('..')
from poi import PoI


# 通过贪婪算法选择行为，默认环境不包含障碍物
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
            actions.append([0, 0])
            continue
        else:
            target = env.pois[target]

        if np.abs(target.x - uav.x) <= uav.v_max and np.abs(target.y - uav.y) <= uav.v_max:  # x和y能直接到达
            dx = target.x - uav.x
            dy = target.y - uav.y
        elif np.abs(target.x - uav.x) <= uav.v_max:  # x 能直接到达
            dx = target.x - uav.x
            dy = uav.v_max if target.y > uav.y else -uav.v_max
        elif np.abs(target.y - uav.y) <= uav.v_max:  # y 能直接到达
            dy = target.y - uav.y
            dx = uav.v_max if target.x > uav.x else -uav.v_max
        else:
            dx = uav.v_max if target.x > uav.x else -uav.v_max
            dy = uav.v_max if target.y > uav.y else -uav.v_max
        actions.append([dx, dy])
    return actions
