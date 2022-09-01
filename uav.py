# --------------------------------------------------------
# 无人机类
# --------------------------------------------------------
import numpy as np

class UAV:
    def __init__(self, x=500, y=500, v_max=20):
        # 位置信息
        self.init_x = x  # 初始横坐标位置
        self.init_y = y  # 初始纵坐标位置
        self.x = x
        self.y = y
        # 速度信息
        self.v_x = 0
        self.v_y = 0
        self.v_max = v_max
        # 电量信息
        self.energy = 1  # 当前电量
        # 观测值
        self.obs = []
        # 无人机颜色
        self.color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

    # 重置无人机状态
    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.v_x = 0
        self.v_y = 0
        self.energy = 1

    # 计算执行某个行为的能量损耗(绝对值)
    def cal_energy_loss(self, action):
        return 0.001*(abs(action[0]+abs(action[1])))

    # 更新速度，并将其修正到合适的区间
    def set_velocity(self, v_x, v_y):
        # x轴速度
        if v_x > self.v_max:
            self.v_x = self.v_max
        elif v_x < -self.v_max:
            self.v_x = -self.v_max
        else:
            self.v_x = v_x
        # y轴速度
        if v_y > self.v_max:
            self.v_y = self.v_max
        elif v_y < -self.v_max:
            self.v_y = -self.v_max
        else:
            self.v_y = v_y
