from base_agent import Agent 
from model.networks import QMix_Nets, GRU_QNets
import torch.nn as nn 
import torch.optim as optim
from utils.exp_replay import ExperienceReplay
from ..model.networks import Pretrained_QNets, MixingNetwork
import numpy as np
import torch 

class QMIXAgent:
    def __init__(self, obs_shape, num_agents, action_dim, lr=1e-3, gamma=0.99, device="cpu"):
        self.device = torch.device(device)
        self.num_agents = num_agents
        self.action_dim = action_dim
        # Shared Q-Network
        self.q_net = Pretrained_QNets(obs_shape, action_dim).to(self.device)
        self.target_q_net = Pretrained_QNets(obs_shape, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Mixing Network
        self.mixing_net = MixingNetwork(num_agents).to(self.device)
        self.target_mixing_net = MixingNetwork(num_agents).to(self.device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        self.target_q_net.eval()
        self.target_mixing_net.eval()

        self.optimizer = optim.Adam(list(self.q_net.parameters()) + list(self.mixing_net.parameters()), lr=lr)
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()

        # epsilon decay 
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.05

    def load_pretrained(self): 
        pass 
    

    def get_action(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                return self.q_net(state_tensor).argmax().item()


    
    def train(self, dataloader):
        self.q_net.train()
        self.mixing_net.train()
    
        for obs, action, reward, next_obs, done in dataloader:
            self.q_net.zero_grad()
            self.mixing_net.zero_grad()
    
            # Đổi thứ tự permute và chuyển vào device
            obs = obs.permute(0, 1, 4, 2, 3).to(self.device)  # [batch_size, num_agents, channels, height, width]
            next_obs = next_obs.permute(0, 1, 4, 2, 3).to(self.device)
    
            action = action.to(self.device)  # [batch_size, num_agents]
            reward = reward.unsqueeze(1).to(self.device)  # [batch_size, num_agents, 1]
            done = done.unsqueeze(1).to(self.device)  # [batch_size, num_agents, 1]
    
            # Xác định agent chết và sống
            dead_agent_mask = (action == -1).float()  # [batch_size, num_agents]
            alive_agent_mask = 1 - dead_agent_mask  # [batch_size, num_agents]
    
            batch_size, num_agents, channels, height, width = obs.shape
    
            # Dự đoán từng agent với q_net
            obs_q_values = []  # Danh sách chứa output của từng agent
            for agent_idx in range(num_agents):
                obs_single_agent = obs[:, agent_idx]  # [batch_size, channels, height, width]
                
                if dead_agent_mask[:, agent_idx].any():  # Nếu agent đã chết
                    q_values_single_agent = torch.full(
                        (batch_size, self.q_net.network[-1].out_features),  # [batch_size, action_dim]
                        fill_value=-100,
                        device=self.device
                    )
                else:  # Nếu agent còn sống
                    q_values_single_agent = self.q_net(obs_single_agent)  # [batch_size, action_dim]
                
                obs_q_values.append(q_values_single_agent)
    
            # Stack lại Q-values từ tất cả các agent
            obs_q_values = torch.stack(obs_q_values, dim=1)  # [batch_size, num_agents, action_dim]
    
            # Lấy giá trị Q tương ứng với action
            current_q_values = obs_q_values.gather(-1, action.unsqueeze(-1).clamp(min=0)).squeeze(-1)  # [batch_size, num_agents]
            current_q_values = current_q_values * alive_agent_mask  # Bỏ qua Q-values của agent chết
    
            with torch.no_grad():
                next_q_values = []
                for agent_idx in range(num_agents):
                    next_obs_single_agent = next_obs[:, agent_idx]  # [batch_size, channels, height, width]
    
                    if dead_agent_mask[:, agent_idx].any():  # Nếu agent đã chết
                        q_values_single_agent = torch.full(
                            (batch_size, self.q_net.network[-1].out_features),  # [batch_size, action_dim]
                            fill_value=0,  # Không tính Q-values cho agent chết
                            device=self.device
                        )
                    else:  # Nếu agent còn sống
                        q_values_single_agent = self.target_q_net(next_obs_single_agent)  # [batch_size, action_dim]p
                       
    
                    next_q_values.append(q_values_single_agent)
    
                # Stack lại Q-values của tất cả các agent
                next_q_values = torch.stack(next_q_values, dim=1)  # [batch_size, num_agents, action_dim]
                max_next_q_values = next_q_values.max(dim=-1)[0]  # [batch_size, num_agents]
                masked_next_q_values = max_next_q_values * alive_agent_mask  # Bỏ qua Q-values của agent chết
               
                # Tính Q-value mục tiêu
                target_q_totals = self.target_mixing_net(masked_next_q_values)  # [batch_size, 1]
                
                
                print(reward.sum(dim=1, keepdim=True).shape)
                targets = reward.sum(dim=1, keepdim=True) + self.gamma * target_q_totals * (1 - done.sum(dim=1, keepdim=True))
                
            # Tính tổng Q-values hiện tại
            current_q_totals = self.mixing_net(current_q_values)
   
            # Tính loss và cập nhật
            loss = self.loss_fn(current_q_totals, targets)
            loss.backward()
            self.optimizer.step()

       
    def update_target_networks(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
