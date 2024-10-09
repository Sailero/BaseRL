import numpy as np
import cv2 as cv
from common.utils import save_expert_data
from env.forklift.isaac_sim_env_client import IsaacSimEnvClient
import os

# 初始化专家数据列表
expert_obs_list, expert_action_list, expert_reward_list, expert_next_obs_list, expert_done_list = [], [], [], [], []
expert_path = "./model/expert_data"
if not os.path.exists(expert_path):
    os.makedirs(expert_path)


env = IsaacSimEnvClient()

# 栈板是否随机
pallet_random = True

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

    # 保存专家数据
    expert_obs_list.append(img / 255)
    expert_action_list.append(action)
    expert_reward_list.append(reward / 100)
    expert_next_obs_list.append(img_ / 255)
    expert_done_list.append(terminated)

    # print("wheel vel: ",info["wheel_velocities"])
    # print("wheel pos: ",info["wheel_positions"])
    # print("base pos: ", info["pos"][0])
    # print("base vel: ", info["vel_x"])
    # print("base rot: ", info["vel_rot"])
    # print("state", img.shape, "reward: ", reward, ", terminated: ", terminated, ", truncated: ", truncated)
    # print(info)
    # print("==========")

    cv.imshow("img", img_ / 255.)

    img = img_
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
        if terminated:
            save_expert_data(expert_path, "expert_obs", expert_obs_list)
            save_expert_data(expert_path, "expert_action", expert_action_list)
            save_expert_data(expert_path, "expert_reward", expert_reward_list)
            save_expert_data(expert_path, "expert_next_obs", expert_next_obs_list)
            save_expert_data(expert_path, "expert_done", expert_done_list)
