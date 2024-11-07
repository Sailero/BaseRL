PPO_CONFIG = {
    "batch_size": 128,
    "gamma": 0.95,
    "max_grad_norm": 0.5,
    "lam": 0.95,
    "eps_clip": 0.2,
    "update_nums": 10,
    "ent_coef": 0.05,

    "actor_hidden_dim": 128,
    "critic_hidden_dim": 128,

    "lr_actor": 2e-5,
    "lr_critic": 1e-4

}

BUFFER_CONFIG = {
    "buffer_size": 64
}
