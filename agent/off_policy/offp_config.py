SAC_CONFIG = {
    "lr_alpha": 5e-5,
    "lr_critic": 1e-4,
    "lr_actor": 2e-5,
    "gamma": 0.95,
    "tau": 0.01,
    "actor_hidden_dim": 128,
    "critic_hidden_dim": 128
}

DDPG_CONFIG = {
    "lr_critic": 1e-4,
    "lr_actor": 2e-5,
    "gamma": 0.95,
    "tau": 0.01,
    "epsilon": 0.1,
    "noise_rate": 0.1,
    "actor_hidden_dim": 128,
    "critic_hidden_dim": 128
}

BUFFER_CONFIG = {
    "buffer_size": 1e5,
    "batch_size": 1024
}

