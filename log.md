#### 2024-10-24训练日志：
1. DCDR_PPO simple     
DCDR的训练参数
    parser.add_argument("--dr_min_ratio", type=float, default=0.8,
                        help="Dynamic rl min reward ratio in rl and gail reward")
    parser.add_argument("--dr_max_ratio", type=float, default=0.99,
                        help="Dynamic rl max reward ratio in rl and gail reward")
    parser.add_argument("--start_episode", type=float, default=1000,
                        help="Start episode for dynamic rl reward ratio increasing")
    parser.add_argument("--end_episode", type=float, default=2000,
                        help="End episode for dynamic rl reward ratio increasing")
    parser.add_argument("--train-episodes", type=int, default=3000,
                        help="number of time steps")（稀疏奖励）

average episode steps 25.0  
agent average returns 9.9   
... saving agent checkpoint ...   
The best reward of agent is 9.95 when episode is 3000   


2. DCDR_PPO simple     
DCDR的训练参数
    parser.add_argument("--dr_min_ratio", type=float, default=0.8,
                        help="Dynamic rl min reward ratio in rl and gail reward")
    parser.add_argument("--dr_max_ratio", type=float, default=1,
                        help="Dynamic rl max reward ratio in rl and gail reward")
    parser.add_argument("--start_episode", type=float, default=1000,
                        help="Start episode for dynamic rl reward ratio increasing")
    parser.add_argument("--end_episode", type=float, default=1500,
                        help="End episode for dynamic rl reward ratio increasing")
    parser.add_argument("--train-episodes", type=int, default=3000,
                        help="number of time steps")（稀疏奖励）

average episode steps 25.0  
agent average returns 15.8 
... saving agent checkpoint ...   
The best reward of agent is 15.8  when episode is 3000   

3. DCDR_PPO simple     
DCDR的训练参数
    # DCDR的训练参数
    parser.add_argument("--dr_min_ratio", type=float, default=0.5,
                        help="Dynamic rl min reward ratio in rl and gail reward")
    parser.add_argument("--dr_max_ratio", type=float, default=0.5,
                        help="Dynamic rl max reward ratio in rl and gail reward")
    parser.add_argument("--start_episode", type=float, default=1000,
                        help="Start episode for dynamic rl reward ratio increasing")
    parser.add_argument("--end_episode", type=float, default=1500,
                        help="End episode for dynamic rl reward ratio increasing")
    parser.add_argument("--train-episodes", type=int, default=3000,
                        help="number of time steps")（稀疏奖励）

average episode steps 25.0
agent average returns 5.3
... saving agent checkpoint ...
The best reward of agent is 7.32 when episode is 3800