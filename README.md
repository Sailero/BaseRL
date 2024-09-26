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
