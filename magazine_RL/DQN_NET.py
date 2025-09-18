import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
import torch.optim as optim
import random
from collections import deque

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 已有数据（保持不变）
seed_value = 1234
np.random.seed(seed_value)
UNIT = 40
MAZE_H = 8
MAZE_W = 8
utilization_ratios_device = 0.1
utilization_ratios_server = 0.1
car_flop = 1.3 * 10**12
car_power = 20
# UAV_flop = 1.3 * 10**12*0.4##0.641 * 10**12
UAV_flop = 0.641 * 10**12
# UAV_power = 20##30
UAV_power = 30
e_flop = 330 * 10**12
e_power = 450
d_fai = 5*10**-29
trans_v_up = 100*1024*1024/4 #553081138.4484484
trans_v_dn = 20*1024*1024/4
p_cm = 0.1
# nums_data = np.array([5, 3, 4, 7, 9])  # 客户端本地数据量
partition_point = [0, 1, 2, 3, 4, 5, 6]

num_img_UAV = 3
num_img_car = 1

device_load = [0.3468e9, 0.3519e9, 2.3408e9, 2.3409e9, 5.3791e9, 9.6951e9, 12.077e9]
server_load = [11.7321e9, 11.727e9, 9.7381e9, 9.738e9, 6.6998e9, 2.3838e9, 0.0019e9]
exchanged_data = [2359296, 2359296, 2359296, 2359296, 1179628, 589824, 294912]
privacy_leak = [0.96122, 0.608901, 0.57954889, 0.593044, 0.535525, 0.007155, 0.054303]

# 转换为NumPy数组
np_partition = np.array(partition_point)
np_device = np.array(device_load)
np_server = np.array(server_load)
np_exchanged = np.array(exchanged_data)
np_privacy = np.array(privacy_leak)

def cost_cal(num_data, v_flop, device_power, partition_index):
    partial_device = np_device[partition_index]
    device_time = partial_device * num_data / (v_flop *utilization_ratios_device)

    partial_server = np_server[partition_index]
    server_time = partial_server * num_data / ( e_flop* utilization_ratios_server + 1e-8)
    # print(f"device_time is {device_time}, server_time is {server_time}, cal_time is {device_time+server_time}")

    feature = np_exchanged[partition_index]
    trans_t_up = feature / trans_v_up * num_data
    # print(f"device_time is {device_time}, server_time is {server_time}, cal_time is {device_time+server_time},trans_t_up is {trans_t_up}")
    energy_cal = ((partial_device * device_power) / v_flop + (
            partial_server * e_power * utilization_ratios_server) / e_flop) * num_data
    energy_trans = num_data * p_cm * trans_t_up
    energy = energy_cal + energy_trans
    # print(f"energy cal is{energy_cal}, trans is {energy_trans}")
    landa_trans = 1
    time_all = device_time + server_time + landa_trans * trans_t_up
    return time_all, energy

class HighLevelAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HighLevelAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.to(device)  # 模型移动到GPU

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)



def train():
    # 初始化模型并移动到GPU
    UAV_agent = HighLevelAgent(len(device_load), len(partition_point)).to(device)
    car_agent = HighLevelAgent(len(device_load), len(partition_point)).to(device)
    # server_net = ServerSAC(len(nums_data) + 1, num_users).to(device)
    # alpha_optimizer = optim.Adam([server_net.log_alpha], lr=0.001)
    num_img_UAV = 3
    num_img_car = 1
    optimizer_UAV = torch.optim.Adam(UAV_agent.parameters(), lr=0.001)
    optimizer_car = torch.optim.Adam(car_agent.parameters(), lr=0.001)
    multi_NET2_history = []
    num_episodes = 5001
    # server_buffer = ReplayBuffer()

    for episode in range(num_episodes):
        # 数据转移到GPU
        state_UAV = torch.tensor(np_device*num_img_UAV, dtype=torch.float32).to(device)
        state_UAV = (state_UAV - state_UAV.min()) / (state_UAV.max() - state_UAV.min())
        
        probs_UAV = UAV_agent(state_UAV)
        action_UAV = torch.multinomial(probs_UAV, 1).item()
        partition_UAV = partition_point[action_UAV]
        partition_UAV_normalized = (partition_UAV - np.min(partition_UAV)) / (np.max(partition_UAV) - np.min(partition_UAV))

        state_car = torch.tensor(np_device*num_img_car, dtype=torch.float32).to(device)
        state_car = (state_car - state_car.min()) / (state_car.max() - state_car.min())
        
        probs_car = car_agent(state_car)
        action_car = torch.multinomial(probs_car, 1).item()
        partition_car = partition_point[action_car]
        partition_car_normalized = (partition_car - np.min(partition_car)) / (np.max(partition_car) - np.min(partition_car))
        # 服务器输入处理
        # server_input = np.concatenate([state_high.cpu().numpy(), [partition_normalized]])
        # servers_power = server_net.act(server_input)

        # 计算奖励
        time_UAV, energy_UAV = cost_cal(num_img_UAV, UAV_flop, UAV_power, action_UAV)
        time_car, energy_car = cost_cal(num_img_car, car_flop, car_power, action_car)
            # total_time += time
        reward_car = -(time_car)*0.4 - (energy_car)*0.3 - 0.3*np_privacy[action_car]
        reward_UAV = -(time_UAV)*0.4 - 0.3*(energy_UAV) - 0.3*np_privacy[action_UAV]


        
        
        # # 存储经验
        # next_state_high = state_high.detach().clone()
        # next_partition_normalized = partition_normalized
        # next_server_input = np.concatenate([next_state_high.cpu().numpy(), [next_partition_normalized]])
        # server_buffer.add(server_input, e_power, reward, next_server_input, False)

        # 更新高层智能体
        log_prob_UAV = torch.log(probs_UAV[action_UAV])
        loss_uav = -log_prob_UAV * reward_UAV
        optimizer_UAV.zero_grad()
        loss_uav.backward(retain_graph=True)
        optimizer_UAV.step()
        
        log_prob_car = torch.log(probs_car[action_car])
        loss_car = -log_prob_car * reward_car
        optimizer_car.zero_grad()
        loss_car.backward(retain_graph=True)
        optimizer_car.step()


        if episode % 10 == 0:
            print(f"episode {episode} UAV action is{action_UAV},car action is {action_car},car reward is {reward_car}")
            history_0.append(reward_UAV)
            history_1.append(reward_car)
            
            # print(f"Episode {episode}: Total Time = {time_car+time_UAV},Total_Energy = {energy_car + energy_UAV} Reward = {reward}")
            # print(f"Episode {episode}: Total Time = {total_time}, Reward = {reward}")
            # multi_NET2_history.append(abs(reward))

    # print("Training history:", multi_NET2_history)

if __name__ == "__main__":
    history_0 = []
    history_1 = []
    train()