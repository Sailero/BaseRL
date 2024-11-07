import numpy as np

FORKLIFT_CONFIG = {
    "pallet_random": True,
    "max_episode_len": 80,
    "action_low": np.array([-1, -0.2]),
    "action_high": np.array([0, 0.2])
}