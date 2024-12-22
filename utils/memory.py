"""
Cài đặt replay buffer cho thuật toán Double Q Learning 
"""
from torch.utils.data import Dataset
import torch 
from collections import deque
import random
import numpy as np


"""
Cài đặt replay buffer cho thuật toán Double Q Learning 
"""
class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.stack(state), np.array(action), np.array(reward), 
                np.stack(next_state), np.array(done))
        
    def __len__(self):
        return len(self.buffer)


    def __getitem__(self, idx): 
        state, action, reward, next_state, done = self.buffer[idx]
        return (
            torch.tensor(state), 
            torch.tensor(action), 
            torch.tensor(reward, dtype = torch.float),
            torch.tensor(next_state), 
            torch.tensor(done, dtype = torch.float32)
        )
    
"""
Cài đặt StateMemory cho thuật toán QMix
"""
class StateMemory(Dataset):
    def __init__(self, capacity, num_agents = 162, grouped_agents = 18):
        self.capacity = capacity
        self.memory = [deque(maxlen=capacity) for _ in range(grouped_agents)]
        self.num_agents = num_agents 
        self.grouped_agents = grouped_agents 

    def push(self, idx, state, action, reward, next_state, done):
        self.memory[idx].append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Lấy ngẫu nhiên một batch thông tin từ bộ nhớ -> batch 
        batch được sample ra là một chuỗi hành đồng 
        """
        
        batch = random.sample(self.memory, batch_size)
        idx, state, action, reward, next_state, done = zip(*batch)
        return (
            np.stack(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.stack(next_state),
            np.array(done, dtype=np.float32)
        )

    def ensemble(self): 
        """
        mở rộng tất cả deque trong self.memory đến chiều dài tối đa bằng cách thêm các giá trị None vào cuối.
        state 
        """
        max_len = max([len(agent_memory) for agent_memory in self.memory])
        min_len = min([len(agent_memory) for agent_memory in self.memory])

        if max_len == min_len: return 
        
        for i in range(self.grouped_agents):
            current_len = len(self.memory[i])
            while current_len < max_len:
                
                self.memory[i].append((None, None, None, None, None))
                current_len += 1

    def __len__(self):
        """
        Trả về độ dài của bộ nhớ.
        """
        return min([len(i) for i in self.memory])

    

    def __getitem__(self, idx):
        """
        Trả về dữ liệu tại một chỉ số cụ thể dưới dạng tensor cho tất cả agents.
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in range(self.grouped_agents):
        
            state, action, reward, next_state, done = self.memory[i][idx]

            state = state if state is not None else np.full_like(self.memory[0][0][0], fill_value=-1)
            action = action if action is not None else -1 
            reward = reward if reward is not None else 0.0
            next_state = next_state if next_state is not None else np.full_like(self.memory[0][0][0], fill_value=-1)
            done = done if done is not None else 1.0 

            # Thêm vào danh sách
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

   
        
        return (
            torch.tensor(states, dtype=torch.float32),      
            torch.tensor(actions, dtype=torch.long),        
            torch.tensor(rewards, dtype=torch.float32),      
            torch.tensor(next_states, dtype=torch.float32),  
            torch.tensor(dones, dtype=torch.float32)        
        )