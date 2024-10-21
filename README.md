# 主要参数（arguments）
scenario-name：环境名称，项目中应选择forklift

max-episode-len：每一轮的最大步数（在项目中这个参数并不需要，可以从环境中自己读取）    
train-episodes：要训练的回合数    
load-pre-model：是否加载先前训练的模型。一般用于在停止训练后，一段时间后可以继续训练。    
policy-type：训练算法类型。目前只支持了PPO。DDPG的接口还没有完善。    

st-buffer-size：PPO训练的重要参数，表示收集多少条数据之后开始训练。训练完成之后就会清空数据，然后继续收集数据，等待下一轮训练。    
update-nums：PPO训练的重要参数，表示一轮训练的次数。比如设置为40就是对这一批数据重复训练40次。

compare：如果为True，会收集所有算法的训练数据并绘制训练曲线，对比效果。使用这个需要保证各个算法训练的episodes相同   
evaluate：如果为False则为训练状态；如果为True则为可视化展示状态。   
display-episodes：每隔display-episodes会print过去的平均reward，并检测训练效果是否提升，如果提升则保存当前模型到model中。

#### 10.12补充
./model/MPE expert data：是测试MPE环境中模仿学习是否有效的数据。  
./model/expert data：是模型读取专家数据的路径区域。数据存储格式是.npy。当前里面的数据是MPE的expert data  
imitation-learning：是否进行模仿学习的接口。在完成模仿学习之后，会自动进行evaluate  
imitation-training-nums：模仿学习训练的次数。每一次训练都是让buffer采样以此后扔给train函数进行训练。因此训练时可能会微调算法训练参数。  
forklift_expert_sample.py：是采集叉车专家数据的程序，独立运行。可以自动将数据存储到./model/expert data中。
                           考虑到会多次采样数据，这里的数据累计采集的。只要不手动删除，就会一直保留之前的数据。
* 在模仿学习中，PPO暂时是拿所有的expert data进行训练，样本个数不等于st-buffer-size。因此训练会特别慢。这个还需要思考有没有改进的地方。
* DDPG还无法模仿学习，存在bug。目前只能用PPO

其他参数相对而言没有那么重要，或者对于大体训练流程与效果并不敏感。


# 算法架构
agent：   
    modules：各个算法的训练组件，包括buffer，AC network   
    policy：各算法的训练主逻辑   
    agent.py：调用各个算法的接口   

common：   
    arguments：存放训练参数   
    utils：runner或者主函数中所需要的函数   

env：   
    forklift：项目环境   
    mpe：多智能体粒子环境，有一个simple的环境可以训练。   
    env.py：调用各个环境的接口   

model：保存各种模型的文件夹

runner：
    runner：DDPG的主要运行逻辑类   
    st_runner：PPO的主要运行逻辑类   

main：运行主函数

# 环境

## poetry
推荐使用 poetry 管理 python 项目，可以很方便的管理依赖，环境。
[poetry 使用说明](https://blog.kyomind.tw/python-poetry/)

环境安装：

```bash
poetry install
```

## 运行

```bash
# 进入 poetry 环境
poetry shell

# 运行 main.py
python main.py
```

## 运行 tensorboard
在项目根目录下

```bash
# 进入 poetry 环境
poetry shell

# 运行 tensorboard
tensorboard --logdir=logs
```

