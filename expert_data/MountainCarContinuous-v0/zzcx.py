import numpy as np


obs = np.load("./expert_obs.npy")
action = np.load("./expert_action.npy")

print(len(obs))
for i in range(1000):
    print(f"position:{obs[i, 0]}, velocity:{obs[i, 1]}, action:{action[i, 0]}")
