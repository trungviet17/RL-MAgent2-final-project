import torch.nn as nn
import torch
import torch.nn.functional as F


class Pretrained_QNets(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)


       
class MixingNetwork(nn.Module):
    def __init__(self, num_agents, embed_dim=32):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.hyper_w1 = nn.Linear(1, num_agents * embed_dim)
        self.hyper_b1 = nn.Linear(1, embed_dim)
        self.hyper_w2 = nn.Linear(1, embed_dim)
        self.hyper_b2 = nn.Linear(1, 1)
        self.embed_dim = embed_dim

    def forward(self, agent_qs, state):
        batch_size = agent_qs.size(0)
        states = torch.ones((batch_size, 1)).to(agent_qs.device)  # Dummy state input for hypernet
    
        # Lấy weight và bias từ hyper-net
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
    
        # Tính toán Mixing Network
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1  # Linear transformation
        hidden = torch.relu(hidden)
        q_total = torch.bmm(hidden, w2) + b2  # Output tổng Q-value
        return q_total.squeeze(-1)

    

class Final_QNets(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            # nn.LayerNorm(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            # nn.LayerNorm(84),
            nn.Tanh(),
        )
        self.last_layer = nn.Linear(84, action_shape)

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        x = self.network(x)
        self.last_latent = x
        return self.last_layer(x)
    

class MyQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()

        # CNN Feature Extractor with Fewer Parameters
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Adaptive Pooling for fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate the flattened dimension
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1).unsqueeze(0)
        dummy_output = self.adaptive_pool(self.cnn(dummy_input))
        flatten_dim = dummy_output.reshape(-1).shape[0]
        # Fully Connected Layers with Fewer Parameters
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Final Layer
        self.last_layer = nn.Linear(128, action_shape)

    def forward(self, x):
        # Input shape: (batch_size, C, H, W)
        assert len(x.shape) == 4, "Input tensor must be 4D (batch_size, C, H, W)"
        
        # Pass through CNN
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        
        # Flatten the features
        x = x.reshape(x.size(0), -1)
        
        # Pass through Fully Connected Layers
        x = self.network(x)
        self.last_latent = x

        # Output action values
        return self.last_layer(x)


