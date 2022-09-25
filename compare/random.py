import numpy as np


# 通过随机算法选择行为，默认环境不包含障碍物
def select_actions(env):
    actions = []
    for uav in env.uavs:
        actions.append(2 * env.uavs[0].v_max * np.random.random(2) - env.uavs[0].v_max)
    return actions
