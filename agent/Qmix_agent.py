from base_agent import Agent 
import torch.nn as nn 
import torch.optim as optim
from ..model.networks import  MixingNetwork, SharedQNetwork, PretrainedQNetwork, Final_QNets, MyQNetwork
import numpy as np
import torch 

class QMIXAgent(Agent):
    def __init__(self, obs_shape, num_agents, action_dim, lr=1e-3, gamma=0.99, device="cpu", qnets_name: str = "shared"):
        self.device = torch.device(device)
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.n_observation = obs_shape
        
        self.init_networks(qnets_name)

        self.mixing_net = MixingNetwork(num_agents).to(self.device)
        self.target_mixing_net = MixingNetwork(num_agents).to(self.device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        self.target_mixing_net.eval()

        self.optimizer = optim.Adam(list(self.q_net.parameters()) + list(self.mixing_net.parameters()), lr=lr)
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.05

    
    def init_networks(self, qnets_name: str): 
        if qnets_name == "pretrained": 
            self.q_net = PretrainedQNetwork(self.n_observation, self.action_dim)
            self.target_q_net = PretrainedQNetwork(self.n_observation, self.action_dim)
            
        elif qnets_name == "final":
            self.q_net = Final_QNets(self.n_observation, self.action_dim)
            self.target_q_net = Final_QNets(self.n_observation, self.action_dim)
        elif qnets_name == "myq":
            self.q_net = MyQNetwork(self.n_observation, self.action_dim)
            self.target_q_net = MyQNetwork(self.n_observation, self.action_dim)
        elif qnets_name == "shared":
            self.q_net = SharedQNetwork(self.n_observation, self.action_dim)
            self.target_q_net = SharedQNetwork(self.n_observation, self.action_dim)
        else:
            raise ValueError("Invalid model name")
        
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

    

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
            reward = reward.to(self.device)  # [batch_size, num_agents, 1]
            done = done.to(self.device)  # [batch_size, num_agents, 1]
    
            # Xác định agent chết và sống
            alive_agent_mask = (action != -1).float()  # [batch_size, num_agents]
        
            # Tính Q-values hiện tại
            obs_flat = obs.view(-1, *obs.shape[2:])  # [batch_size * num_agents, channels, height, width]
            obs_q_values = self.q_net(obs_flat).view(obs.size(0), obs.size(1), -1)  # [batch_size, num_agents, action_dim]
        
            current_q_values = obs_q_values.gather(-1, action.unsqueeze(-1).clamp(min=0)).squeeze(-1)  # [batch_size, num_agents]
            current_q_values = current_q_values * alive_agent_mask  # Bỏ qua Q-values của agent chết

            # Tính Q-values mục tiêu
            with torch.no_grad():
                next_obs_flat = next_obs.view(-1, *next_obs.shape[2:])  # [batch_size * num_agents, channels, height, width]
                next_q_values = self.target_q_net(next_obs_flat).view(next_obs.size(0), next_obs.size(1), -1)  # [batch_size, num_agents, action_dim]
    
                max_next_q_values = next_q_values.max(dim=-1)[0]  # [batch_size, num_agents]
                masked_next_q_values = max_next_q_values * alive_agent_mask  # Bỏ qua Q-values của agent chết
                
                target_q_totals = self.target_mixing_net(masked_next_q_values, next_obs)  # [batch_size, 1]
                targets = reward.mean(dim=1, keepdim=True) + self.gamma * target_q_totals * (1 - done.mean(dim=1, keepdim=True))
            
            # Tính tổng Q-values hiện tại
            current_q_totals = self.mixing_net(current_q_values, obs)  # [batch_size, 1]
    
            # Tính loss và cập nhật
            loss = self.loss_fn(current_q_totals, targets)
            loss.backward()
            self.optimizer.step()


       
    def update_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
