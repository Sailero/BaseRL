# 定义策略类型和对应的类映射关系
POLICY_MAP = {
    'DDPG': {
        'policy': 'agent.off_policy.DDPG.ddpg.DDPG',
        'buffer': 'agent.off_policy.replay_buffer.Buffer',
        'online_policy': False
    },
    'SAC': {
        'policy': 'agent.off_policy.SAC.sac_1q.SAC',
        'buffer': 'agent.off_policy.replay_buffer.Buffer',
        'online_policy': False
    },
    'PPO': {
        'policy': 'agent.on_policy.PPO.ppo.PPO',
        'buffer': 'agent.on_policy.replay_buffer.Buffer',
        'online_policy': True
    },
    'GAIL_PPO': {
        'policy': 'agent.imitation_learning.GAIL.gail.GAIL',
        'buffer': 'agent.on_policy.replay_buffer.Buffer',
        'online_policy': True
    },
    'GAIL_SAC': {
        'policy': 'agent.imitation_learning.GAIL.gail.GAIL',
        'buffer': 'agent.off_policy.replay_buffer.Buffer',
        'online_policy': False
    }
}