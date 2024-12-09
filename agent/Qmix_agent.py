from base_agent import Agent 
from model.networks import QMix_Nets, GRU_QNets
import torch.nn as nn 
import torch.optim as optim
from utils.exp_replay import ExperienceReplay
import numpy as np
import torch 

class QMix_Agent(Agent): 

    def __init__(self, num_agent:int, n_observation, n_actions, state_shape,  batch_size, lr, gamma, device = None):
        self.num_agent = num_agent
        self.n_observation = n_observation
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.device = device


        self.q_agent = GRU_QNets(n_observation, n_actions)
        self.q_target_agent = GRU_QNets(n_observation, n_actions)

        self.qmix = QMix_Nets(state_shape, n_actions, num_agent)
        self.qmix_target = QMix_Nets(state_shape, n_actions, num_agent)

        self.q_target_agent.load_state_dict(self.q_agent.state_dict())
        self.qmix_target.load_state_dict(self.qmix.state_dict())


        self.opimizer = optim.Adam(list(self.q_agent.parameters()) + list(self.qmix.parameters()), lr = lr)

        self.buffer = ExperienceReplay(10000)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1



    def get_action(self, observation):
        """
        Input: 
            observations : actions cua mot agent 
        Output: 
            action : int - hành động được chọn 
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else : 
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                return self.q_agent(obs_tensor).argmax().item()
            
     
    
    def train(self): 
        """
        Thuc hien cap nhat lai tham so cua network 
        """
        if len(self.buffer) < self.batch_size:
            return
        
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch 

        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        




        pass 