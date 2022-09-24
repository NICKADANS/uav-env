# --------------------------------------------------------
# 无人机类
# --------------------------------------------------------
from copy import deepcopy

import numpy as np

class UAV:
    def __init__(self, x=500, y=500, v_max=10, color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))):
        # 位置信息
        self.init_x = deepcopy(x)  # 初始横坐标位置
        self.init_y = deepcopy(y)  # 初始纵坐标位置
        self.x = deepcopy(x)
        self.y = deepcopy(y)
        # 速度信息
        self.v_max = v_max
        # 电量信息
        self.energy = 1  # 当前电量
        # 观测值
        self.view_range = v_max * 2 + 1  # 视野范围
        self.obs = []
        # 无人机颜色
        self.color = color

    # 重置无人机状态
    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.obs = []
        self.energy = 1

    # 计算执行某个行为的能量损耗(绝对值)
    def cal_energy_loss(self, action):
        return 0.005
