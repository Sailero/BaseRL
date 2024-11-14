import gym
import numpy as np


class GymEnv:
    def __init__(self, config):
        # 创建环境
        if config.env.name in ['MountainCarContinuous-v0', 'Pendulum-v1']:
            self.scenario_name = config.env.name
            if config.task_type == 'evaluate':
                self.env = gym.make(config.env.name, render_mode='human')
            else:
                self.env = gym.make(config.env.name)

            # 获取观测信息
            self.agent_obs_dim = list(self.env.observation_space.shape)

            # 获取动作信息
            self.agent_action_dim = self.env.action_space.shape[0]  # 动作空间的维度
            self.action_low = self.env.action_space.low  # 动作空间的最小值
            self.action_high = self.env.action_space.high  # 动作空间的最大值

            # 获取环境运行信息
            self.max_episode_len = self.env.spec.max_episode_steps

        else:
            raise ValueError(f"{config.env.name} not supported")

    def reset(self):
        obs = self.env.reset()
        return obs[0]

    def step(self, action):
        # 反归一化
        action = action * self.action_high
        obs_, reward, done, info1, info2 = self.env.step(action)
        if self.scenario_name == "MountainCarContinuous-v0":
            reward = reward + np.sum(action) ** 2 * 0.1
        return obs_, reward, done, [info1, info2]

    def render(self):
        self.env.render()


class MpeEnv:
    def __init__(self, config):
        # 创建环境
        if config.env.name in ['simple']:
            from env.mpe.make_env import make_env
            self.env = make_env(config.env.name)

            # 获取观测信息
            self.agent_obs_dim = list(self.env.observation_space[0].shape)

            # 获取动作信息
            self.agent_action_dim = self.env.action_space[0].n
            self.action_high = 1
            self.action_low = -1

            # 获取环境运行信息
            self.max_episode_len = 25

        else:
            raise ValueError(f"{config.env.name} not supported")

    def reset(self):
        obs = self.env.reset()
        return obs[0]

    def step(self, action):
        # 反归一化
        action = (np.array(action) * self.action_high).tolist()
        action_n = [action]
        obs_, reward, done, info = self.env.step(action_n)
        return obs_[0], reward[0], done[0], info

    def render(self):
        self.env.render()


class Env:
    def __init__(self, config):
        # 创建环境
        if config.env.name in ['MountainCarContinuous-v0', 'Pendulum-v1']:
            self.env = GymEnv(config)

        elif config.env.name in ['simple']:
            from env.mpe.make_env import make_env
            self.env = MpeEnv(config)

        else:
            raise ValueError(f"{config.env.name} not supported")

        # 获取观测信息
        self.agent_obs_dim = self.env.agent_obs_dim

        # 获取动作信息
        self.agent_action_dim = self.env.agent_action_dim
        self.action_low = self.env.action_low
        self.action_high = self.env.action_high

        # 获取环境运行信息
        self.max_episode_len = self.env.max_episode_len

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs_, reward, done, info = self.env.step(action)
        return obs_, reward, done, info

    def render(self):
        self.env.render()
