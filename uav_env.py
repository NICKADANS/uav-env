# --------------------------------------------------------
# 搭建UAV的强化学习环境
# --------------------------------------------------------
import common
import cv2
import numpy as np
from uav import UAV
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
        self.init_pois = pois
        self.init_uavs = uavs
        self.pois = self.init_pois
        self.uavs = self.init_uavs
        self.obstacles = obstacles

    # 重置图片
    def reset(self):
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.image[:, :] = (255, 255, 255)
        self.draw_pois(self.init_pois)
        self.draw_uavs(self.init_uavs)
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
        cv2.circle(self.image, (int(uav.x), int(uav.y)), 2, uav.color, -1)

    # 绘制障碍物
    def draw_obs(self, obstacles):
        for obs in obstacles:
            cv2.circle(self.image, (int(obs[0]), int(obs[1])), 1, common.OBS_COLOR, -1)


class ObservationSpace:
    def __init__(self, pois, obstacles, uavs):
        # 状态空间维度
        self.dim = len(pois) * 3 + len(obstacles) * 2 + len(uavs) * 3  # poi的信息，障碍物的信息，无人机的信息
        # 状态空间内容
        self.observations = []
        # 各个观测值在向量中的起点
        self.poi_index = 0
        self.obs_index = len(pois) * 3
        self.uav_index = self.obs_index + len(obstacles) * 2
        # 初始化观测空间
        for poi in pois:
            self.observations.extend(poi.to_list())
        for obstacle in obstacles:
            self.observations.extend(obstacle)
        for uav in uavs:
            l = [uav.x, uav.y, uav.energy]
            self.observations.extend(l)

    # 更新状态空间中的兴趣点观测值
    def update_pois_obs(self, pois):
        idx = self.poi_index
        for poi in pois:
            self.observations[idx+2] = poi.done
            idx += 3
        return self.observations[self.poi_index: self.obs_index]

    # 更新状态空间中的无人机状态值
    def update_uavs_obs(self, uavs):
        idx = self.uav_index
        for uav in uavs:
            self.observations[idx] = uav.x
            idx += 1
            self.observations[idx] = uav.y
            idx += 1
            self.observations[idx] = uav.energy
            idx += 1
        return self.observations[self.uav_index:]


class ActionSpace:
    def __init__(self, uavs):
        # 行为空间维度
        self.dim = len(uavs) * 2  # 需要得到所有无人机的速度
        # 行为空间内容:所有UAV的速度
        self.actions = []
        for uav in uavs:
            l = [uav.v_x, uav.v_y]
            self.actions.extend(l)


class UavEnvironment:
    def __init__(self, pois, obstacles, uav_num):
        # 初始化障碍物/兴趣点/无人机，保存初始兴趣点状态
        self.init_pois = pois
        self.pois = self.init_pois
        self.obstacles = obstacles
        self.uavs = [UAV() for i in range(uav_num)]
        # 初始化观测空间和行为空间，保存初始观测值
        self.obsvervation_space = ObservationSpace(self.pois, self.obstacles, self.uavs)
        self.action_space = ActionSpace(self.uavs)
        self.init_obs = self.obsvervation_space.observations
        # 为每个无人机初始化观测值
        for uav in self.uavs:
            uav.obs = self.obsvervation_space.observations
        # 初始化渲染
        self.render = UavEnvRender(pois=self.pois, obstacles=self.obstacles)
        self.is_render = False # 默认不开启渲染
        # 环境是否共享奖励值，默认为共享
        self.share_reward = True

    # 重置环境状态
    def reset(self):
        # 重置兴趣点和观测值
        self.pois = self.init_pois
        self.obsvervation_space.observations = self.init_obs
        # 重置每个无人机
        for uav in self.uavs:
            uav.reset()
            uav.obs = self.obsvervation_space.observations
        # 重置渲染
        self.render.reset()
        return self.obsvervation_space.observations

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
        # 倘若共享奖励值
        reward = np.sum(reward_n)
        if self.share_reward:
            reward_n = [reward for _ in range(len(self.uavs))]
        # 环境状态值
        new_state_n = [self.obsvervation_space.observations for _ in range(len(self.uavs))]
        return new_state_n, reward_n, done_n, info_n

    # 为环境里的单个无人机执行行为
    def _step(self, uav, action):
        reward = 0
        # 判断是否有电执行下一步动作
        if uav.energy > uav.cal_energy_loss(action):
            # 扣除本次行为的电量
            uav.energy -= uav.cal_energy_loss(action)
            # 计算无人机新的坐标
            new_x = int(uav.x + action[0])
            new_y = int(uav.y + action[1])
            # 判断无人机执行行为后的状态，并计算奖励
            if 0 <= new_x < 1000 and 0 <= new_y < 1000:  # 无人机位于界内
                # 计算奖励
                reward = -(abs(action[0]) + abs(action[1]))
                # 判断是否采集了某个兴趣点
                raidus = 15
                for poi in self.pois:
                    if (poi.x - new_x)**2 + (poi.y - new_y)**2 <= raidus**2 and poi.done == 0:
                        reward = 100
                        poi.done = 1
                        # 绘制poi
                        self.render.draw_poi(poi)
                        break
                # 判断是否撞到了障碍物
                for obstacle in self.obstacles:
                    if obstacle[0] == new_x and obstacle[1] == new_y:
                        reward = -100
                        break
                # 更新该无人机的位置
                uav.x = new_x
                uav.y = new_y
                # 判断是否撞到了无人机
                raidus = 2
                for uav in self.uavs:
                    dis = (new_x - uav.x)**2 + (new_y - uav.y)**2
                    if 1e-6 < dis <= raidus**2:
                        reward = -100
                        break
                # 更新环境观测值
                self.obsvervation_space.update_pois_obs(self.pois)
                self.obsvervation_space.update_uavs_obs(self.uavs)
                # 更新无人机的观测值
                uav.obs = self.obsvervation_space.observations
            else:  # 无人机位于界外
                # 计算奖励
                reward = -999
                # 更新该无人机的位置
                if new_x < 0:
                    uav.x = 0
                if new_y < 0:
                    uav.y = 0
                if new_x >= 1000:
                    uav.x = 999
                if new_y >= 1000:
                    uav.y = 999
                # 更新环境的观测值
                self.obsvervation_space.update_uavs_obs(self.uavs)
                # 更新无人机观测值
                uav.obs = self.obsvervation_space.observations
        else:  # 没电执行下一步动作
            uav.energy = 0
        # 重置无人机的速度
        uav.v_x = 0
        uav.v_y = 0
        # 渲染无人机的新位置
        self.render.draw_uav(uav)
        return uav.obs, action, reward, None


if __name__ == "__main__":
    pois = np.load("Data/pois.npy", allow_pickle=True)
    obstacles = np.load("Data/obstacles.npy")
    env = UavEnvironment(pois, obstacles, 3)
    env.reset()
    while True:
        actions = []
        '''
        action = v_x和v_y 属于 [-uav.v_max, uav.v_max]
        '''
        for uav in env.uavs:
            actions.extend(2 * uav.v_max * np.random.random(2) - uav.v_max)
        obs, rewards, dones, _ = env.step(actions)
        cv2.imshow("env", env.render.image)
        cv2.waitKey(0)
        done = 1
        for d in dones:
            if d == 0:
                done = 0
                break
        if done ==  1:
            break
    # print(env._get_obs(uavs[0]))
    # obsn, rewn, donen, _ = env.step(actions)
