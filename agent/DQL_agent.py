from agent.base_agent import Agent
import numpy as np 
import torch 
from model.networks import PretrainedQNetwork
import torch.nn as nn 
import torch.optim as optim

class DQNAgent(Agent):
    def __init__(self, observation_shape, action_shape, batch_size=64, lr=1e-3, gamma=0.6, device="cpu"):
        self.device = torch.device(device)
        self.q_net = PretrainedQNetwork(observation_shape, action_shape).float().to(self.device)
        self.target_net = PretrainedQNetwork(observation_shape, action_shape).float().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.action_shape = action_shape
        self.epsilon = 1.0
        self.epsilon_decay = 0.97
        self.epsilon_min = 0.05
        self.loss_fn = nn.MSELoss()
    

    def get_action(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_shape)
        else:
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                return self.q_net(state_tensor).argmax().item()

    def train(self, dataloader):
        """
            cap nhat lai tham so mo hinh voi input dau vao 
        """
        self.q_net.train()
        for obs, action, reward, next_obs, done in dataloader: 
            self.q_net.zero_grad()
    
            obs = obs.permute(0, 3, 1, 2).to(self.device) 
            action = action.unsqueeze(1).to(self.device)
            reward = reward.unsqueeze(1).to(self.device)
            next_obs = next_obs.to(self.device)
            next_obs = next_obs.permute(0, 3, 1, 2).to(self.device)
            done = done.unsqueeze(1).to(self.device)
    
            # cap nhat gia tri q 
            with torch.no_grad(): 
                target_q_values = reward + self.gamma * (1 - done) * self.target_net(next_obs).max(1, keepdim=True)[0]
    
            q_values = self.q_net(obs).gather(1, action)
    
            loss = self.loss_fn(q_values, target_q_values)
            loss.backward()
            self.optimizer.step()
       

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


