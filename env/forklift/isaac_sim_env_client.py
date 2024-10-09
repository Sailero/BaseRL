from random import random

import numpy as np
from env.forklift.env_client import EnvClient
import cv2 as cv
from common.utils import save_expert_data
import time


class IsaacSimEnvClient(EnvClient):
    def __init__(self, ip: str = "192.168.7.117",
                 port: int = 11800):
        super().__init__(ip, port)

        # state 图像尺寸 640x480
        self.observation_space = (480, 640)
        # 动作空间 2维 [车体速度 m/s，车体的角速度 rad/s]
        self.action_space = (2,)

        self.img = None

    def reset(self, random=False):
        """
        重置环境
        :param random: 栈板位置随机
        :return: tuple[state: np.ndarray, info: dict]
        """
        # 发送重置指令
        self.img, info = self.send({"cmd": "reset", "random": random})
        return self.img, info

    def step(self, action: np.ndarray):
        """

        :param action: 动作指令 [车体速度 m/s，车体的角速度 rad/s]
        :return: tuple[state: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict]
            state: 图像数据 640x480
            reward: 奖励值
            terminated: 是否终止
            truncated: 是否截断
            info: 其他信息
                {'vel_x': array([ 4.4052392e-02, -1.4687826e-01, -6.7848123e-06], dtype=float32),
                 'vel_rot': array([ 1.0735348e-05, -1.4843585e-04,  1.8175127e-02], dtype=float32),
                 'pos': (array([-0.76966435, -1.5114584 ,  0.00157279], dtype=float32),
                         array([-7.5684786e-01,  5.5972487e-05,  6.3955085e-05,  6.5359116e-01], dtype=float32))}
        """
        # 发送动作指令
        self.img, reward, terminated, truncated, info = self.send({"cmd": "step", "action": action})
        return self.img, reward, terminated, truncated, info

    def render(self):
        cv.imshow("img", self.img / 255.)
        key = cv.waitKey(1)
        if key & 0xFF == ord('s'):
            cv.waitKey(0)


if __name__ == "__main__":
    env = IsaacSimEnvClient()

    # 栈板是否随机
    pallet_random = False

    img, info = env.reset(random=pallet_random)
    print("info: ", info)
    cv.imshow("img", img / 255.)
    cv.waitKey(0)

    # 不按按键 车辆停止
    is_stop = True
    key_action = np.array([0., 0.])
    # 线速度 m/s
    vel_x = 0.4
    # 角度 rad
    rot = 10. * 0.017453292

    while True:
        if is_stop:
            action = np.array([0., 0.])
        else:
            action = key_action
            # action = np.random.rand(2)
            # action = 2 * action - 1
            # action[0] = 2 * action[0] - 1
            # print(action)

        # 发送动作指令
        img_, reward, terminated, truncated, info = env.step(action)

        # print("wheel vel: ",info["wheel_velocities"])
        # print("wheel pos: ",info["wheel_positions"])
        # print("base pos: ", info["pos"][0])
        # print("base vel: ", info["vel_x"])
        # print("base rot: ", info["vel_rot"])
        # print("state", img.shape, "reward: ", reward, ", terminated: ", terminated, ", truncated: ", truncated)
        # print(info)
        # print("==========")

        cv.imshow("img", img_ / 255.)
        # 按键控制
        key = cv.waitKey(0)

        is_stop = True
        if key & 0xFF == ord('t'):
            # 退出
            break
        elif key & 0xFF == ord('r'):
            # 重置
            _, info = env.reset(random=pallet_random)
            print("info: ", info)
        elif key & 0xFF == ord('p'):
            # 暂停 （仿真不运行）
            cv.waitKey(0)
        elif key & 0xFF == ord('w'):
            # 前进
            is_stop = False
            key_action = np.array([vel_x, 0.])
        elif key & 0xFF == ord('a'):
            # 左转
            is_stop = False
            key_action = np.array([0., -rot])
        elif key & 0xFF == ord('q'):
            # 左前
            is_stop = False
            key_action = np.array([vel_x, -rot])
        elif key & 0xFF == ord('d'):
            # 右转
            is_stop = False
            key_action = np.array([0., rot])
        elif key & 0xFF == ord('e'):
            # 右前
            is_stop = False
            key_action = np.array([vel_x, rot])
        elif key & 0xFF == ord('s'):
            # 后退
            is_stop = False
            key_action = np.array([-vel_x, 0.])
        elif key & 0xFF == ord('z'):
            # 左后
            is_stop = False
            key_action = np.array([-vel_x, rot])
        elif key & 0xFF == ord('c'):
            # 右后
            is_stop = False
            key_action = np.array([-vel_x, -rot])

        if terminated or truncated:
            _, info = env.reset(random=pallet_random)
            print("terminated: ", terminated, ", truncated: ", truncated)
            print("info: ", info)
