import numpy as np


# obs = np.load("../MPE expert data/expert_obs.npy")
# action = np.load("../MPE expert data/expert_action.npy")
# reward = np.load("../MPE expert data/expert_reward.npy")
# obs_ = np.load("../MPE expert data/expert_next_obs.npy")
# done = np.load("../MPE expert data/expert_done.npy")
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

# print(obs)
# print(action)
# print(obs_)
# print(reward)
# print(done)
print(np.sum(done))

