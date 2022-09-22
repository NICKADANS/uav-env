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
    for uav in env.uavs:
        mindistance = 2000  # 最短距离
        poi1 = PoI(0, 0)  # 保存POI
        for poi in env.pois:
            diatance = np.sqrt((uav.x - poi.x) ** 2 + (uav.y - poi.y) ** 2)
            if poi.done == 0 and diatance < mindistance:
                mindistance = diatance
                poi1 = poi
        if np.abs(poi1.x - uav.x) <= uav.v_max and np.abs(poi1.y - uav.y) <= uav.v_max:  # x和y能直接到达
            poi1.done = 1
            dx = poi1.x - uav.x
            dy = poi1.y - uav.y
        elif np.abs(poi1.x - uav.x) <= uav.v_max:  # x 直接到达
            dx = poi1.x - uav.x
            dy = uav.v_max if poi1.y > uav.y else -uav.v_max
        elif np.abs(poi1.y - uav.y) <= uav.v_max:  # y 直接到达
            dy = poi1.y - uav.y
            dx = uav.v_max if poi1.x > uav.x else -uav.v_max
        else:
            dx = uav.v_max if poi1.x > uav.x else -uav.v_max
            dy = uav.v_max if poi1.y > uav.y else -uav.v_max
        actions.append([dx, dy])
    return actions
