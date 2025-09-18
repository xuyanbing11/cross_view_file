
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.optim as optim

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
d_flop = 1.3 * 10**12
d_power = 20
car_flop = 1.3 * 10**12*0.5
car_power = 20
e_flop = 330 * 10**12
e_power = 450
d_fai = 5*10**-29
trans_v_up = 100*1024*1024/4 #553081138.4484484
trans_v_dn = 20*1024*1024/4
p_cm = 0.1
num_users = 5
# nums_data = np.array([5, 3, 4, 7, 9])  # 客户端本地数据量
partition_point = [1, 2, 3, 4, 5, 6, 7]

# device_load = [
#     18.874e6, 988.874e6, 1064.371e6, 2002.371e6,
#     2077.868e6, 4843.868e6, 4919.365e6, 5833.365e6  # 设备端计算量
# ]

# server_load = [
#     5814.506e6, 4844.506e6, 4769.009e6, 3831.009e6,  # 服务器端计算量
#     3755.512e6, 989.512e6, 914.015e6, 0.015e6
# ]

# exchanged_data = [
#     12582912, 12582912, 6291456, 6291456,  # 模型大小
#     3145728, 3145728, 1572864, 1572864
# ]
device_load = [0.3468e9, 0.3519e9, 2.3408e9, 2.3409e9, 5.3791e9, 9.6951e9, 12.077e9]
server_load = [11.7321e9, 11.727e9, 9.7381e9, 9.738e9, 6.6998e9, 2.3838e9, 0.0019e9]
exchanged_data = [2359296, 2359296, 2359296, 2359296, 1179628, 589824, 294912]
# 转换为NumPy数组
np_partition = np.array(partition_point)
np_device = np.array(device_load)
np_server = np.array(server_load)
np_exchanged = np.array(exchanged_data)



def cost_cal(num_data, server_power, partition_index):
    partial_device = np_device[partition_index]
    device_time = partial_device * num_data / (d_flop *utilization_ratios_device)

    partial_server = np_server[partition_index]
    server_time = partial_server * num_data / ( e_flop* utilization_ratios_server + 1e-8)
    # print(f"device_time is {device_time}, server_time is {server_time}, cal_time is {device_time+server_time}")

    feature = np_exchanged[partition_index]
    trans_t_up = feature / trans_v_up * num_data
    # print(f"device_time is {device_time}, server_time is {server_time}, cal_time is {device_time+server_time},trans_t_up is {trans_t_up}")
    energy_cal = ((partial_device * d_power) / d_flop + (
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

# class ServerSAC(nn.Module):
#     def __init__(self, input_dim, num_devices):
#         super().__init__()
#         self.num_devices = num_devices

#         # 特征提取
#         self.feature_net = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.LayerNorm(512),
#             nn.SiLU(),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.SiLU()
#         )

#         # 均值和标准差网络
#         self.mean_net = nn.Linear(256, num_devices)
#         self.log_std_net = nn.Linear(256, num_devices)

#         # Q网络
#         self.q1 = nn.Sequential(
#             nn.Linear(input_dim + num_devices, 512),
#             nn.LayerNorm(512),
#             nn.SiLU(),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.SiLU(),
#             nn.Linear(256, 1)
#         )

#         self.q2 = nn.Sequential(
#             nn.Linear(input_dim + num_devices, 512),
#             nn.LayerNorm(512),
#             nn.SiLU(),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.SiLU(),
#             nn.Linear(256, 1)
#         )

#         # 目标网络
#         self.q1_target = nn.Sequential(
#             nn.Linear(input_dim + num_devices, 512),
#             nn.LayerNorm(512),
#             nn.SiLU(),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.SiLU(),
#             nn.Linear(256, 1)
#         )

#         self.q2_target = nn.Sequential(
#             nn.Linear(input_dim + num_devices, 512),
#             nn.LayerNorm(512),
#             nn.SiLU(),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.SiLU(),
#             nn.Linear(256, 1)
#         )

#         # 复制参数并固定目标网络
#         for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
#             target_param.data.copy_(param.data)
#         for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
#             target_param.data.copy_(param.data)

#         # 熵参数
#         self.log_alpha = torch.zeros(1, requires_grad=True).to(device)
#         self.target_entropy = -num_devices

#         # 优化器
#         self.policy_optimizer = optim.Adam(
#             list(self.feature_net.parameters()) +
#             list(self.mean_net.parameters()) +
#             list(self.log_std_net.parameters()), lr=0.001)
#         self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=0.001)
        
#         # 模型移动到GPU
#         self.to(device)

#     def forward(self, state):
#         features = self.feature_net(state)
#         mean = self.mean_net(features)
#         log_std = self.log_std_net(features)
#         log_std = torch.clamp(log_std, -20, 2)
#         return mean, log_std

#     def act(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 数据移到GPU
#         with torch.no_grad():
#             mean, log_std = self.forward(state)
#             std = log_std.exp()
#             dist = Normal(mean, std)
#             action = dist.rsample()
#             action_probs = torch.sigmoid(action)
#             normalized_action = action_probs / action_probs.sum(dim=1, keepdim=True)
#         return normalized_action.squeeze(0).cpu().numpy()  # 结果移回CPU

# class ReplayBuffer:
#     def __init__(self, capacity=10000):
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0

#     def add(self, state, action, reward, next_state, done):
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(None)
#         self.buffer[self.position] = (state, action, reward, next_state, done)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         indices = np.random.choice(len(self.buffer), batch_size, replace=False)
#         states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
#         # 转换为GPU张量
#         states = torch.FloatTensor(np.array(states)).to(device)
#         actions = torch.FloatTensor(np.array(actions)).to(device)
#         rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
#         next_states = torch.FloatTensor(np.array(next_states)).to(device)
#         dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
#         return states, actions, rewards, next_states, dones

#     def __len__(self):
#         return len(self.buffer)

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
    num_episodes = 8000
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
        total_time = 0
        
        time_UAV, energy_UAV = cost_cal(num_img_UAV, e_power, action_UAV)
        time_car, energy_car = cost_cal(num_img_car, e_power, action_car)
            # total_time += time
        reward = -(time_car+time_UAV)*0.4 - (energy_car + energy_UAV)*0.3

        # # 存储经验
        # next_state_high = state_high.detach().clone()
        # next_partition_normalized = partition_normalized
        # next_server_input = np.concatenate([next_state_high.cpu().numpy(), [next_partition_normalized]])
        # server_buffer.add(server_input, e_power, reward, next_server_input, False)

        # 更新高层智能体
        log_prob_UAV = torch.log(probs_UAV[action_UAV])
        loss_uav = -log_prob_UAV * reward
        optimizer_UAV.zero_grad()
        loss_uav.backward(retain_graph=True)
        optimizer_UAV.step()
        
        log_prob_car = torch.log(probs_car[action_car])
        loss_car = -log_prob_car * reward
        optimizer_car.zero_grad()
        loss_car.backward(retain_graph=True)
        optimizer_car.step()

        # 更新服务器智能体
#         if len(server_buffer) > 64:
#             states, actions, rewards, next_states, dones = server_buffer.sample(64)
            
#             # 计算Q值
#             features = server_net.feature_net(states)
#             mean, log_std = server_net.mean_net(features), server_net.log_std_net(features)
#             std = log_std.exp()
#             dist = Normal(mean, std)
#             z = dist.rsample()
#             new_actions = torch.sigmoid(z)
#             normalized_actions = new_actions / new_actions.sum(dim=1, keepdim=True)
            
#             # 目标Q值计算
#             with torch.no_grad():
#                 target_features = server_net.feature_net(next_states)
#                 target_mean, target_log_std = server_net.mean_net(target_features), server_net.log_std_net(target_features)
#                 target_std = target_log_std.exp()
#                 target_dist = Normal(target_mean, target_std)
#                 target_z = target_dist.rsample()
#                 target_actions = torch.sigmoid(target_z)
#                 target_normalized_actions = target_actions / target_actions.sum(dim=1, keepdim=True)
                
#                 alpha = server_net.log_alpha.exp()
#                 entropy = -0.5 * (1 + 2 * log_std + torch.log(2 * torch.tensor(np.pi))).sum(dim=1, keepdim=True)
                
#                 q1_input = torch.cat([next_states, target_normalized_actions], dim=1)
#                 q2_input = torch.cat([next_states, target_normalized_actions], dim=1)
#                 q1_target = server_net.q1_target(q1_input)
#                 q2_target = server_net.q2_target(q2_input)
#                 min_q_target = torch.min(q1_target, q2_target)
#                 target_q = rewards + 0.99 * (1 - dones) * (min_q_target - alpha * entropy)
            
#             # 更新Q网络
#             q1 = server_net.q1(torch.cat([states, normalized_actions], dim=1))
#             q2 = server_net.q2(torch.cat([states, normalized_actions], dim=1))
#             q1_loss = F.mse_loss(q1, target_q)
#             q2_loss = F.mse_loss(q2, target_q)
#             q_loss = q1_loss + q2_loss
            
#             server_net.q_optimizer.zero_grad()
#             q_loss.backward()
#             server_net.q_optimizer.step()
            
#             # 更新策略网络
#             policy_loss = -(torch.min(server_net.q1(torch.cat([states, normalized_actions], dim=1),
#                                    server_net.q2(torch.cat([states, normalized_actions], dim=1)) - alpha * entropy).mean()
            
#             server_net.policy_optimizer.zero_grad()
#             policy_loss.backward()
#             server_net.policy_optimizer.step()
            
#             # 更新alpha
#             alpha_loss = -(server_net.log_alpha * (entropy.detach() + server_net.target_entropy)).mean()
#             alpha_optimizer.zero_grad()
#             alpha_loss.backward()
#             alpha_optimizer.step()
            
#             # 软更新目标网络
#             for target_param, param in zip(server_net.q1_target.parameters(), server_net.q1.parameters()):
#                 target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
#             for target_param, param in zip(server_net.q2_target.parameters(), server_net.q2.parameters()):
#                 target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Time = {time_car+time_UAV},Total_Energy = {energy_car + energy_UAV} Reward = {reward}")
            # print(f"Episode {episode}: Total Time = {total_time}, Reward = {reward}")
            multi_NET2_history.append(abs(reward))

    print("Training history:", multi_NET2_history)

if __name__ == "__main__":
    train()