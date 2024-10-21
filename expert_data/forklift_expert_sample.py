import numpy as np
import cv2 as cv
from common.utils import save_expert_data
from env.forklift.isaac_sim_env_client import IsaacSimEnvClient
import os

# 初始化专家数据列表
expert_obs_list, expert_action_list, expert_reward_list, expert_next_obs_list, expert_done_list = [], [], [], [], []
expert_path = "./forklift"
if not os.path.exists(expert_path):
    os.makedirs(expert_path)

# 栈板是否随机
env = IsaacSimEnvClient(pallet_random=True)

state, info = env.reset()
print("info: ", info)
# cv.imshow("img", state / 255.0)
# cv.waitKey(0)

# 不按按键 车辆停止
is_stop = True
key_action = np.array([0., 0.])
# 线速度 m/s
vel_x = 0.4
# 角度 rad
rot = 10. * 0.017453292

# 是否一开始就记录数据
auto_record = False
print("auto_record: ", auto_record)

# 开始记录数据标记(用于采集特殊的场景数据)
is_record = auto_record

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
    state_next, reward, terminated, truncated, info = env.step(action)

    if is_record:
        # 保存专家数据
        expert_obs_list.append(state)
        expert_next_obs_list.append(state_next)
        expert_action_list.append(action)
        expert_reward_list.append(reward / 100.)
        expert_done_list.append(terminated)

    # print("wheel vel: ",info["wheel_velocities"])
    # print("wheel pos: ",info["wheel_positions"])
    # print("base pos: ", info["pos"][0])
    # print("base vel: ", info["vel_x"])
    # print("base rot: ", info["vel_rot"])
    # print("state", img.shape, "reward: ", reward, ", terminated: ", terminated, ", truncated: ", truncated)
    # print(info)
    # print("==========")

    cv.imshow("img", state_next / 255.0)

    state = state_next
    # 按键控制
    key = cv.waitKey(0)

    is_stop = True
    if key & 0xFF == ord('t'):
        # 退出
        break
    elif key & 0xFF == ord('r'):
        # 重置
        _, info = env.reset()
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
        key_action = np.array([0.1, -rot])
    elif key & 0xFF == ord('q'):
        # 左前
        is_stop = False
        key_action = np.array([vel_x, -rot])
    elif key & 0xFF == ord('d'):
        # 右转
        is_stop = False
        key_action = np.array([0.1, rot])
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
    elif key & 0xFF == ord('x'):
        # 左后(快速转)
        is_stop = False
        key_action = np.array([-0.1, rot * 3])
    elif key & 0xFF == ord('c'):
        # 右后
        is_stop = False
        key_action = np.array([-vel_x, -rot])
    elif key & 0xFF == ord('v'):
        # 右后(快速转)
        is_stop = False
        key_action = np.array([-0.1, -rot * 3])
    elif key & 0xFF == ord('k'):
        # 开始记录数据/停止记录数据
        is_record = not is_record
        print("is_record: ", is_record)
    elif key & 0xFF == ord('u'):
        # 反转 auto_record
        auto_record = not auto_record
        print("auto_record: ", auto_record)
    elif key & 0xFF == ord('l'):
        # 结束并保存数据
        terminated = True
        print("manual terminated!")
    elif key & 0xFF == ord('o'):
        # 结束并丢弃数据
        truncated = True
        print("manual truncated!")


    if terminated or truncated:
        print("before reset, info: ", info)
        state, info = env.reset()
        is_record = auto_record
        action = np.array([0., 0.])
        print("terminated: ", terminated, ", truncated: ", truncated)
        print("info: ", info)
        if terminated:
            save_expert_data(expert_path, "expert_obs", expert_obs_list)
            save_expert_data(expert_path, "expert_action", expert_action_list)
            save_expert_data(expert_path, "expert_reward", expert_reward_list)
            save_expert_data(expert_path, "expert_next_obs", expert_next_obs_list)
            save_expert_data(expert_path, "expert_done", expert_done_list)

        expert_obs_list.clear()
        expert_action_list.clear()
        expert_reward_list.clear()
        expert_next_obs_list.clear()
        expert_done_list.clear()
