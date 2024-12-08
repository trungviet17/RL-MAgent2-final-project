from base_agent import Agent
from utils.exp_replay import ExperienceReplay
from torch.optim import Adam
import numpy as np 
import torch 
from model.pretrained_model import QNetwork
import torch.nn as nn 

class DQLAgent(Agent):

    def __init__(self, n_observation, n_actions, buffer_size, batch_size, gamma, lr, target_update_freq):
        super().__init__(n_observation, n_actions)

        self.qnetwork = QNetwork(n_observation, n_actions)
        self.target_qnetwork = QNetwork(n_observation, n_actions)
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.target_qnetwork.eval()


        self.buffer = ExperienceReplay(buffer_size)
        self.optimizer = Adam(self.qnetwork.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update_freq = target_update_freq


        # epsilon decay 
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05 



    def get_action(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_shape)
        else:
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                return self.qnetwork(state_tensor).argmax().item()


    def train(self):
        """
        training the agent 
        """

        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        valid_indices = [i for i in range(len(actions)) if actions[i] is not None]
        if not valid_indices:  
            return

        states = np.array([states[i] for i in valid_indices])
        actions = np.array([actions[i] for i in valid_indices])
        rewards = np.array([rewards[i] for i in valid_indices])
        next_states = np.array([next_states[i] for i in valid_indices])
        dones = np.array([dones[i] for i in valid_indices])

        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)


        q_values = self.qnetwork(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_qnetwork(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # t8nh loss
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)