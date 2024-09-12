#开发者：Bright Fang
#开发时间：2022/6/28 23:02
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from matplotlib import pyplot as plt
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
env = gym.make('Pendulum-v1').unwrapped
'''Pendulum环境状态特征是三个，杆子的sin(角度)、cos（角度）、角速度，（状态是无限多个，因为连续），动作值是力矩，限定在[-2,2]之间的任意的小数，所以是连续的（动作也是无限个）'''
state_number=env.observation_space.shape[0]
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
torch.manual_seed(0)#如果你觉得定了随机种子不能表达代码的泛化能力，你可以把这两行注释掉
# env.seed(0)
RENDER=False
EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0003
BATCH = 4
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization
Switch=0
'''由于PPO也是基于A-C框架，所以我把PPO的编写分为两部分，PPO的第一部分 Actor'''
'''PPO的第一步  编写A-C框架的网络，先编写actor部分的actor网络，actor的网络有新与老两个网络'''
class ActorNet(nn.Module):
    def __init__(self,inp,outp):
        super(ActorNet, self).__init__()
        self.in_to_y1=nn.Linear(inp,100)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.out=nn.Linear(100,outp)
        self.out.weight.data.normal_(0,0.1)
        self.std_out = nn.Linear(100, outp)
        self.std_out.weight.data.normal_(0, 0.1)
    '''生成均值与标准差，PPO必须这样做，一定要生成分布（所以需要mean与std），不然后续学习策略里的公式写不了，DDPG是可以不用生成概率分布的'''
    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        mean=max_action*torch.tanh(self.out(inputstate))#输出概率分布的均值mean
        std=F.softplus(self.std_out(inputstate))#softplus激活函数的值域>0
        return mean,std
'''再编写critic部分的critic网络，PPO的critic部分与AC算法的critic部分是一样，PPO不一样的地方只在actor部分'''
class CriticNet(nn.Module):
    def __init__(self,input,output):
        super(CriticNet, self).__init__()
        self.in_to_y1=nn.Linear(input,100)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.out=nn.Linear(100,output)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        Q=self.out(inputstate)
        return Q
class Actor():
    def __init__(self):
        self.old_pi,self.new_pi=ActorNet(state_number,action_number),ActorNet(state_number,action_number)#这只是均值mean
        self.optimizer=torch.optim.Adam(self.new_pi.parameters(),lr=A_LR,eps=1e-5)
    '''第二步 编写根据状态选择动作的函数'''
    def choose_action(self,s):
        # print('s', s)
        inputstate = torch.FloatTensor(s)
        mean,std=self.old_pi(inputstate)
        dist = torch.distributions.Normal(mean, std)
        action=dist.sample()
        action=torch.clamp(action,min_action,max_action)
        action_logprob=dist.log_prob(action)
        return action.detach().numpy(),action_logprob.detach().numpy()
    '''第四步  actor网络有两个策略（更新old策略）————————把new策略的参数赋给old策略'''
    def update_oldpi(self):
        self.old_pi.load_state_dict(self.new_pi.state_dict())
    '''第六步 编写actor网络的学习函数，采用PPO2，即OpenAI推出的clip形式公式'''
    def learn(self,bs,ba,adv,bap):
        bs = torch.FloatTensor(bs)
        ba = torch.FloatTensor(ba)
        adv = torch.FloatTensor(adv)
        bap = torch.FloatTensor(bap)
        for _ in range(A_UPDATE_STEPS):
            mean, std = self.new_pi(bs)
            dist_new=torch.distributions.Normal(mean, std)
            action_new_logprob=dist_new.log_prob(ba)
            ratio=torch.exp(action_new_logprob - bap.detach())
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - METHOD['epsilon'], 1 + METHOD['epsilon']) * adv
            loss = -torch.min(surr1, surr2)
            loss=loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.new_pi.parameters(), 0.5)
            self.optimizer.step()
class Critic():
    def __init__(self):
        self.critic_v=CriticNet(state_number,1)#改网络输入状态，生成一个Q值
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=C_LR,eps=1e-5)
        self.lossfunc = nn.MSELoss()
    '''第三步  编写评定动作价值的函数'''
    def get_v(self,s):
        inputstate = torch.FloatTensor(s)
        return self.critic_v(inputstate)
    '''第五步  计算优势——————advantage，后面发现第五步计算出来的adv可以与第七步合为一体，所以这里的代码注释了，但是，计算优势依然算是可以单独拎出来的一个步骤'''
    # def get_adv(self,bs,br):
    #     reality_v=torch.FloatTensor(br)
    #     v=self.get_v(bs)
    #     adv=(reality_v-v).detach()
    #     return adv
    '''第七步  编写actor-critic的critic部分的learn函数，td-error的计算代码（V现实减去V估计就是td-error）'''
    def learn(self,bs,br):
        bs = torch.FloatTensor(bs)
        reality_v = torch.FloatTensor(br)
        for _ in range(C_UPDATE_STEPS):
            v=self.get_v(bs)
            td_e = self.lossfunc(reality_v, v)
            self.optimizer.zero_grad()
            td_e.backward()
            nn.utils.clip_grad_norm_(self.critic_v.parameters(), 0.5)
            self.optimizer.step()
        return (reality_v-v).detach()
if Switch==0:
    print('PPO2训练中...')
    actor=Actor()
    critic=Critic()
    all_ep_r = []
    for episode in range(EP_MAX):
        observation = env.reset()[0] #环境重置
        buffer_s, buffer_a, buffer_r,buffer_a_logp = [], [], [],[]
        reward_totle=0
        for timestep in range(EP_LEN):
            if RENDER:
                env.render()
            action,action_logprob=actor.choose_action(observation)
            observation_, reward, done, info1, info2 = env.step(action)
            observation_ = observation_
            buffer_s.append(observation)
            buffer_a.append(action)
            buffer_r.append((reward+8)/8)    # normalize reward, find to be useful
            buffer_a_logp.append(action_logprob)
            observation=observation_
            reward_totle+=reward
            reward = (reward - reward.mean()) / (reward.std() + 1e-5)

         #PPO 更新
            if (timestep+1) % BATCH == 0 or timestep == EP_LEN-1:
                v_observation_ = critic.get_v(observation_)
                discounted_r = []
                for reward in buffer_r[::-1]:
                    v_observation_ = reward + GAMMA * v_observation_
                    discounted_r.append(v_observation_.detach().numpy())
                discounted_r.reverse()
                bs, ba, br,bap = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r),np.vstack(buffer_a_logp)
                buffer_s, buffer_a, buffer_r,buffer_a_logp = [], [], [],[]
                advantage=critic.learn(bs,br)#critic部分更新
                actor.learn(bs,ba,advantage,bap)#actor部分更新
                actor.update_oldpi()  # pi-new的参数赋给pi-old
                # critic.learn(bs,br)
        if episode == 0:
            all_ep_r.append(reward_totle)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + reward_totle * 0.1)
        print("\rEp: {} |rewards: {}".format(episode, reward_totle), end="")
        #保存神经网络参数
        if episode % 50 == 0 and episode > 100:#保存神经网络参数
            save_data = {'net': actor.old_pi.state_dict(), 'opt': actor.optimizer.state_dict(), 'i': episode}
            torch.save(save_data, ".\PPO2_model_actor.pth")
            save_data = {'net': critic.critic_v.state_dict(), 'opt': critic.optimizer.state_dict(), 'i': episode}
            torch.save(save_data, ".\PPO2_model_critic.pth")
    env.close()
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()
else:
    print('PPO2测试中...')
    aa=Actor()
    cc=Critic()
    checkpoint_aa = torch.load(".\PPO2_model_actor.pth")
    aa.old_pi.load_state_dict(checkpoint_aa['net'])
    checkpoint_cc = torch.load(".\PPO2_model_critic.pth")
    cc.critic_v.load_state_dict(checkpoint_cc['net'])
    for j in range(10):
        state = env.reset()[0]
        total_rewards = 0
        for timestep in range(EP_LEN):
            env.render()
            # time.sleep(0.05)
            action, action_logprob = aa.choose_action(state)
            new_state, reward, done, info1, info2 = env.step(action)  # 执行动作
            total_rewards += reward
            state = new_state
        print("Score：", total_rewards)
    env.close()