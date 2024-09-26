import numpy as np

from env_client import EnvClient


class IsaacSimEnvClient(EnvClient):
    def __init__(self, ip: str = "192.168.7.117",
                 port: int = 11800):
        super().__init__(ip, port)

        # state 图像尺寸 640x480
        self.observation_space = (640, 480)
        # 动作空间 2维 [车体速度 m/s，车体的角速度 rad/s]
        self.action_space = (2,)

    def reset(self):
        """
        重置环境

        :return: tuple[state: np.ndarray, info: dict]
        """
        # 发送重置指令
        return self.send({"cmd": "reset"})

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
        return self.send({"cmd": "step", "action": action})


if __name__ == "__main__":
    import cv2 as cv

    env = IsaacSimEnvClient()

    img, info = env.reset()
    print("info: ", info)
    cv.imshow("img", img / 255.)
    cv.waitKey(0)

    # 车辆停止但是仿真运行
    is_stop = False

    while True:
        if is_stop:
            action = np.array([0., 0.])
        else:
            action = np.array([-0.4, 0.])
        # 发送动作指令
        img, reward, terminated, truncated, info = env.step(action)
        # print("wheel vel: ",info["wheel_velocities"])
        # print("wheel pos: ",info["wheel_positions"])
        # print("base pos: ", info["pos"][0])
        # print("base vel: ", info["vel_x"])
        # print("base rot: ", info["vel_rot"])
        print("reward: ", reward, ", terminated: ", terminated, ", truncated: ", truncated)
        print("==========")
        cv.imshow("img", img / 255.)
        # quit if q is pressed
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            cv.waitKey(0)
        elif key & 0xFF == ord('x'):
            is_stop = not is_stop
