# --------------------------------------------------------
# 搭建UAV的强化学习环境
# --------------------------------------------------------
from copy import deepcopy

import common
import cv2
import numpy as np
from uav import UAV

from compare import greedy, random


from poi import PoI


class UavEnvRender:
    # 初始化，根据输入值生成一张空白图
    def __init__(self, height=common.DEFAULT_HEIGHT, width=common.DEFAULT_WIDTH, pois=[], obstacles=[], uavs=[]):
        self.height = height
        self.width = width
        # 生成一张默认大小为 1000x1000 的空白图
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # 调整BGR值，将图设为白色
        self.image[:, :] = (255, 255, 255)
        # 兴趣点，障碍物及无人机
        self.pois = deepcopy(pois)
        self.uavs = deepcopy(uavs)
        self.init_pois = deepcopy(pois)
        self.init_uavs = deepcopy(uavs)

        self.obstacles = obstacles

    # 重置图片
    def reset(self):
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.image[:, :] = (255, 255, 255)
        self.pois = deepcopy(self.init_pois)
        self.uavs = deepcopy(self.init_uavs)
        self.draw_pois(self.pois)
        self.draw_uavs(self.uavs)
        self.draw_obs(self.obstacles)

    # 绘制兴趣点
    def draw_pois(self, pois):
        for poi in pois:
            self.draw_poi(poi)

    # 更新兴趣点状态
    def draw_poi(self, poi):
        if poi.done == 1:
            cv2.circle(self.image, (int(poi.x), int(poi.y)), 3, common.POI_COLOR_OVER, -1)
        else:
            cv2.circle(self.image, (int(poi.x), int(poi.y)), 3, common.POI_COLOR_GATHER, -1)

    # 绘制无人机
    def draw_uavs(self, uavs):
        for uav in uavs:
            self.draw_uav(uav)

    # 更新无人机状态
    def draw_uav(self, uav):
        cv2.circle(self.image, (int(uav.x), int(uav.y)), 3, uav.color, -1)

    # 绘制障碍物
    def draw_obs(self, obstacles):
        for obs in obstacles:
            cv2.circle(self.image, (int(obs[0]), int(obs[1])), common.OBS_RADIUS, common.OBS_COLOR, -1)


# 对于每个Agent的观测空间
class ObservationSpace:
    def __init__(self, pois, obstacles, uavs):
        # 状态空间维度
        self.dim = len(pois) * 1 + len(obstacles) * 1 + len(uavs) * 1  # poi的信息，障碍物的信息，无人机的信息


# 对于每个Agent的行为空间
class ActionSpace:
    def __init__(self, uavs):
        # 行为空间维度
        self.dim = 2  # 无人机的速度


class UavEnvironment:
    def __init__(self, pois, obstacles, uav_num):
        # 初始化障碍物/兴趣点/无人机，保存初始兴趣点状态
        self.pois = deepcopy(pois)
        self.obstacles = obstacles
        self.uavs = [UAV(color= common.UAV_COLOR[i]) for i in range(uav_num)]
        self.init_pois = deepcopy(pois)
        # 初始化观测空间和行为空间，保存初始观测值
        self.obsvervation_space = ObservationSpace(self.pois, self.obstacles, self.uavs)
        self.action_space = ActionSpace(self.uavs)
        # 初始化渲染
        self.render = UavEnvRender(pois=self.pois, obstacles=self.obstacles)
        self.is_render = True # 默认开启渲染
        # 环境是否共享奖励值，默认为共享
        self.share_reward = True
        # 初始化环境
        self.reset()

    # 重置环境状态
    def reset(self):
        # 重置兴趣点和观测值
        self.pois = deepcopy(self.init_pois)
        # 重置每个无人机
        for uav in self.uavs:
            uav.reset()
        # 重置渲染
        if self.is_render:
            self.render.reset()
        return self.cal_env_obs()

    # 执行行为
    def step(self, actions):
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        # 为每个无人机执行行为
        for uav in self.uavs:
            _, _, reward, _ = self._step(uav, actions[i])
            reward_n.append(reward)
            i += 1
        # 计算是否完成
        for uav in self.uavs:
            if uav.energy != 0:
                done_n.append(0)
            else:
                done_n.append(1)
        # 计算
        new_state_n = self.cal_env_obs()
        # 倘若共享奖励值
        reward = np.sum(reward_n)
        if self.share_reward:
            reward_n = [reward for _ in range(len(self.uavs))]
        # 环境状态值
        return new_state_n, reward_n, done_n, info_n

    # 为环境里的单个无人机执行行为
    def _step(self, uav, action):
        reward = 0
        # 判断是否有电执行下一步动作
        if uav.energy >= uav.cal_energy_loss(action):
            # 扣除本次行为的电量
            uav.energy -= uav.cal_energy_loss(action)
            # 计算无人机新的坐标
            new_x = uav.x + action[0]
            new_y = uav.y + action[1]
            # 判断无人机执行行为后的状态，并计算奖励
            if 0 <= new_x < 1000 and 0 <= new_y < 1000:  # 无人机位于界内
                # 计算奖励
                reward = -0.01
                # 判断是否采集了某个兴趣点
                radius = uav.v_max
                for poi in self.pois:
                    if (poi.x - new_x)**2 + (poi.y - new_y)**2 <= radius**2 and poi.done == 0:
                        reward = 10
                        poi.done = 1
                        # 绘制poi
                        if self.is_render:
                            self.render.draw_poi(poi)
                        break
                # 判断是否撞到了障碍物
                radius = common.OBS_RADIUS
                for obstacle in self.obstacles:
                    if (obstacle[0] - new_x)**2 + (obstacle[1] - new_y)**2 <= radius**2:
                        reward = -100
                        uav.energy = 0
                        break
                # 更新该无人机的位置
                uav.x = new_x
                uav.y = new_y

            else:  # 无人机位于界外
                # 计算奖励
                reward = -100
                uav.energy = 0
                # 更新该无人机的位置
                if new_x < 0:
                    uav.x = 0
                if new_y < 0:
                    uav.y = 0
                if new_x >= 1000:
                    uav.x = 999
                if new_y >= 1000:
                    uav.y = 999
        else:  # 没电执行下一步动作
            uav.energy = 0
        # 渲染无人机的新位置
        self.render.draw_uav(uav)
        return uav.obs, action, reward, None

    # 估计执行一步action所得的奖励
    def cal_reward(self, action):
        total_reward = 0
        for i, uav in enumerate(self.uavs):
            # 没电执行下一步动作
            if uav.energy < uav.cal_energy_loss(action):
                reward = 0
            else:
                new_x = uav.x + action[i][0]
                new_y = uav.y + action[i][1]
                # 如果无人机下一步位于界内
                if 0 <= new_x < 1000 and 0 <= new_y < 1000:
                    # 如果无人机什么都没做
                    reward = -0.01
                    radius = uav.v_max
                    # 如果无人机在采集兴趣点
                    for poi in self.pois:
                        if (poi.x - new_x)**2 + (poi.y - new_y)**2 <= radius**2 and poi.done == 0:
                            reward = 10
                            break
                    # 如果无人机撞到障碍物
                    radius = common.OBS_RADIUS
                    for obstacle in self.obstacles:
                        if (obstacle[0] - new_x) ** 2 + (obstacle[1] - new_y) ** 2 <= radius ** 2:
                            reward = -100
                            uav.energy = 0
                            break
                    # pass
                # 如果无人机位于界外
                else:
                    reward = -100
            total_reward += reward
        return total_reward

    def cal_env_obs(self):
        for uav in self.uavs:
            if len(uav.obs) == 0:
                uav.obs = [0 for _ in range(self.obsvervation_space.dim)]
            # 更新PoI的观测值
            i = 0
            for p in self.pois:
                uav.obs[i] = np.sqrt((uav.x - p.x)**2 + (uav.y - p.y)**2)
                i += 1
            # 更新障碍物的观测值
            for obs in self.obstacles:
                np.sqrt((uav.x - obs[0])**2 + (uav.y - obs[1])**2)
                i += 1
            # 更新无人机的观测值
            for u in self.uavs:
                uav.obs[i] = np.sqrt((uav.x - u.x)**2 + (uav.y - u.y)**2)
                i += 1
        for o in uav.obs:
            o /= 1000.0
        return [uav.obs for uav in self.uavs]

if __name__ == "__main__":
    pois = np.load("data/pois.npy", allow_pickle=True)
    # obstacles = np.load("data/obstacles.npy")
    obstacles = [[650, 650], [300, 400]]
    env = UavEnvironment(pois, obstacles, 3)
    for i in range(0, 100):
        env.reset()
        gameover = False
        while not gameover:
            # actions = []
            # '''
            # action = v_x和v_y 属于 [-uav.v_max, uav.v_max]
            # '''
            # for uav in env.uavs:
            #     actions.append(2 * np.random.random(2) - 1)
            actions = greedy.select_actions(env)
            obs, rewards, dones, _ = env.step(actions.numpy())
            # 判断游戏是否结束
            gameover = True
            for d in dones:
                if d == 0:
                    gameover = False

            if gameover:
                cv2.imshow("env", env.render.image)
                cv2.waitKey(0)
                break
        count = 0
        for p in env.pois:
            if p.done == 1:
                count += 1
        print(count)

    # print(env._get_obs(uavs[0]))
    # obsn, rewn, donen, _ = env.step(actions)
