import numpy as np


obs = np.load("expert_obs.npy")
action = np.load("expert_action.npy")
reward = np.load("expert_reward.npy")
obs_ = np.load("expert_next_obs.npy")
done = np.load("expert_done.npy")

def check_nan_inf(obs):
    if np.isnan(obs).any():
        print("Warning: Input contains NaN values.")
        return True
    if np.isinf(obs).any():
        print("Warning: Input contains Inf values.")
        return True
    return False

def check_value_range(obs, min_threshold=-1e5, max_threshold=1e5):
    if np.min(obs) < min_threshold or np.max(obs) > max_threshold:
        print(f"Warning: Input values are outside the range [{min_threshold}, {max_threshold}].")
        return True
    return False

print(np.min(obs))