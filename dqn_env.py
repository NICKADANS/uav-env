# --------------------------------------------------------
# 搭建UAV的强化学习环境
# --------------------------------------------------------
from copy import deepcopy

import numpy

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
        # 生成一张图用于训练
        self.train_image = deepcopy(self.image)

    # 重置图片
    def reset(self):
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.image[:, :] = (255, 255, 255)
        self.pois = deepcopy(self.init_pois)
        self.uavs = deepcopy(self.init_uavs)
        self.draw_pois(self.pois)
        self.draw_uavs(self.uavs)
        self.draw_obs(self.obstacles)
        self.train_image = deepcopy(self.image)

    # 绘制兴趣点
    def draw_pois(self, pois):
        for poi in pois:
            self.draw_poi(poi)

    # 更新兴趣点状态
    def draw_poi(self, poi):
        if poi.done == 1:
            cv2.circle(self.image, (int(poi.x), int(poi.y)), 5, common.POI_COLOR_OVER, -1)
            cv2.circle(self.train_image, (int(poi.x), int(poi.y)), 5, common.POI_COLOR_OVER, -1)
        else:
            cv2.circle(self.image, (int(poi.x), int(poi.y)), 5, common.POI_COLOR_GATHER, -1)
            cv2.circle(self.train_image, (int(poi.x), int(poi.y)), 5, common.POI_COLOR_GATHER, -1)

    # 绘制无人机
    def draw_uavs(self, uavs):
        for uav in uavs:
            self.draw_uav(uav)

    # 更新无人机状态
    def draw_uav(self, uav):
        cv2.circle(self.image, (int(uav.x), int(uav.y)), 4, uav.color, -1)

    # 绘制障碍物
    def draw_obs(self, obstacles):
        for obs in obstacles:
            cv2.circle(self.image, (int(obs[0]), int(obs[1])), common.OBS_RADIUS, common.OBS_COLOR, -1)

    # 绘制当前训练图像
    def draw_train_img(self, uavs):
        img = deepcopy(self.train_image)
        for uav in uavs:
            cv2.circle(img, (int(uav.x), int(uav.y)), 4, uav.color, -1)
        return img

# 对于每个Agent的观测空间
class ObservationSpace:
    def __init__(self, uavs):
        # 状态空间维度
        self.dim = (3, uavs[0].view_range, uavs[0].view_range)


# 对于每个Agent的行为空间
class ActionSpace:
    def __init__(self, uavs):
        # 行为空间维度
        self.n = 9  # 无人机的速度
        self.actions = [(-20, 0), (20, 0), (0, 0), (0, 20), (0, -20), (10, 10), (-10, 10), (10, -10), (-10, -10)]

    def sample(self):
        indices = np.random.choice(len(self.actions))
        return indices


class UavEnvironment:
    def __init__(self, pois, obstacles, uav_num, uav_init_pos=[]):
        # 初始化障碍物/兴趣点/无人机，保存初始兴趣点状态
        self.pois = deepcopy(pois)
        self.obstacles = obstacles
        if len(uav_init_pos) != uav_num:
            self.uavs = [UAV(color=common.UAV_COLOR[i]) for i in range(uav_num)]
        else:
            self.uavs = [UAV(
                x=uav_init_pos[i][0], y=uav_init_pos[i][1], color=common.UAV_COLOR[i]) for i in range(uav_num)
            ]
        self.init_pois = deepcopy(pois)
        # 系统收集的兴趣点个数
        self.poi_done = 0

        # 初始化观测空间和行为空间，保存初始观测值
        self.obsvervation_space = ObservationSpace(self.uavs)
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
        self.poi_done = 0
        # 重置每个无人机
        for uav in self.uavs:
            uav.reset()
        # 重置渲染
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
            reward = self._step(uav, actions[i])
            reward_n.append(reward)
            i += 1
        # 计算是否完成
        for uav in self.uavs:
            if uav.energy != 0:
                done_n.append(0)
            else:
                done_n.append(1)
        # 计算新的观测值
        new_state_n = self.cal_env_obs()
        # 倘若共享奖励值
        reward = np.sum(reward_n)
        if self.share_reward:
            reward_n = [reward for _ in range(len(self.uavs))]
        # 环境状态值
        return new_state_n, reward_n, done_n, info_n

    # 为环境里的单个无人机执行行为
    def _step(self, uav, action):
        # 渲染无人机的新位置
        self.render.draw_uav(uav)
        reward = 0
        # 判断是否有电执行下一步动作
        if uav.energy >= uav.cal_energy_loss(action):
            # 扣除本次行为的电量
            uav.energy -= uav.cal_energy_loss(action)
            # 计算无人机新的坐标
            new_x = uav.x + self.action_space.actions[action][0]
            new_y = uav.y + self.action_space.actions[action][1]
            # 判断无人机执行行为后的状态，并计算奖励
            if 0 <= new_x < 1000 and 0 <= new_y < 1000:  # 无人机位于界内
                # 判断是否采集了某个兴趣点
                radius = 20
                mindis = 2000
                for poi in self.pois:
                    if poi.done == 0:
                        dis = np.sqrt((poi.x - new_x)**2 + (poi.y - new_y)**2)
                        mindis = dis if dis < mindis else mindis
                        if dis <= radius :
                            reward += 5
                            poi.done = 1
                            self.poi_done += 1
                            mindis = 0
                            # 绘制poi
                            self.render.draw_poi(poi)
                reward -= mindis * 0.001
                # # 判断是否在其他无人机附近
                # radius = 2 * uav.v_max
                # reward += 10  # 新的位置在自身原来的位置附近
                # for uav in self.uavs:
                #     dis = np.sqrt((uav.x - new_x)**2 + (uav.y - new_y)**2)
                #     if dis <= radius:
                #         reward -= 10
                # 判断是否撞到了障碍物
                radius = common.OBS_RADIUS
                for obstacle in self.obstacles:
                    if (obstacle[0] - new_x)**2 + (obstacle[1] - new_y)**2 <= radius**2:
                        reward -= 100
                        uav.energy = 0
                        break
                # 更新该无人机的位置
                uav.x = new_x
                uav.y = new_y

            else:  # 无人机位于界外
                # 计算奖励
                reward -= 100
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
        return reward

    # 估计执行一步action所得的奖励
    def cal_reward(self, action):
        total_reward = 0
        for i, uav in enumerate(self.uavs):
            # 没电执行下一步动作
            reward = 0
            if uav.energy >= uav.cal_energy_loss(action):
                new_x = uav.x + self.action_space.actions[action][0]
                new_y = uav.y + self.action_space.actions[action][1]
                # 如果无人机下一步位于界内
                if 0 <= new_x < 1000 and 0 <= new_y < 1000:
                    # 如果无人机什么都没做
                    reward = 0
                    mindis = 2000
                    radius = 20
                    # 如果无人机在采集兴趣点
                    for poi in self.pois:
                        dis = np.sqrt((poi.x - new_x) ** 2 + (poi.y - new_y) ** 2)
                        mindis = dis if dis < mindis else mindis
                        if dis <= radius:
                            reward += 5
                            mindis = 0
                    reward -= mindis * 0.001
                    # 判断是否在其他无人机附近
                    # radius = 2 * uav.v_max
                    # reward += 10  # 新的位置在自身原来的位置附近
                    # for uav in self.uavs:
                    #     dis = np.sqrt((uav.x - new_x) ** 2 + (uav.y - new_y) ** 2)
                    #     if dis <= radius:
                    #         reward -= 10
                    # 如果无人机撞到障碍物
                    radius = common.OBS_RADIUS
                    for obstacle in self.obstacles:
                        if (obstacle[0] - new_x) ** 2 + (obstacle[1] - new_y) ** 2 <= radius ** 2:
                            reward -= 100
                            break
                # 如果无人机位于界外
                else:
                    reward -= 100
            total_reward += reward
        return total_reward

    # 计算环境归一化后的观测值
    def cal_env_obs(self):
        img = self.render.draw_train_img(self.uavs)
        for uav in self.uavs:
            uav.obs = np.zeros((uav.view_range, uav.view_range, 3), dtype=np.float)
            # 观测区域：一个 view_range * view_range 的正方形区域
            x_left = int(uav.x - uav.view_range//2)
            y_top = int(uav.y - uav.view_range//2)
            for i in range(0, uav.view_range):
                for j in range(0, uav.view_range):
                    if 0 <= x_left + i < 1000 and 0 <= y_top + j < 1000:
                        uav.obs[j, i] = img[y_top + j, x_left + i] / 255.0
            # cv2.imshow("s", uav.obs)
            # cv2.waitKey(0)

            uav.obs = np.transpose(uav.obs, (2, 0, 1))
        return np.array([uav.obs for uav in self.uavs])


if __name__ == "__main__":
    pois = np.load("data/pois.npy", allow_pickle=True)
    # obstacles = np.load("data/obstacles.npy")
    obstacles = [[650, 650], [300, 400]]
    n_agents = 2
    env = UavEnvironment(pois, obstacles, n_agents)
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
            actions = []
            for i in range(n_agents):
                action = env.action_space.sample()
                actions.append(action)
            obs, rewards, dones, _ = env.step(actions)
            print(rewards)
            # print(obs)
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
