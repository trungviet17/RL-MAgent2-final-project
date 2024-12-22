import torch.nn as nn
import torch
import torch.nn.functional as F


def kaiming_init(m):
    """
    Khởi tạo tham số của lớp theo Kaiming Initialization.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  
        if m.bias is not None:
            nn.init.zeros_(m.bias)  

"""
Đây là kiến trúc pretrained được sử dụng cho red.pt 
"""
class PretrainedQNetwork(nn.Module):
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
       # self.apply(kaiming_init)

    def forward(self, x):
        assert len(x.shape) >= 3
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)
    

"""
Đây là kiến trúc được cài đặt trong final_red.pt 
"""
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
    

"""
Đây là kiến trúc mạng sử đổi (được sử dụng trong thí nghiệm 2)
"""
class MyQNetwork(nn.Module):
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
       # self.apply(kaiming_init)

    def forward(self, x):
        assert len(x.shape) >= 3
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)

"""
Đây là kiến trúc mạng dùng chung giữa các Agent trong thuật toán QMix - tương tự như mạng Pretrained 
"""
class SharedQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(SharedQNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3, padding=1),
            nn.ReLU(),
        )
        
        dummy_input = torch.randn(1, observation_shape[-1], observation_shape[0], observation_shape[1])
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]

        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_shape)
        )
        self.apply(kaiming_init)

    def forward(self, x):

        if len(x.shape) == 4: 
            x = x.permute(0, 3, 1, 2) 
        elif len(x.shape) == 3: 
            x = x.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W

        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        return self.network(x)


       
"""
Đây là cài đặt kiến trúc mạng Mixing trong thuật toán QMix
"""
class MixingNetwork(nn.Module):
    def __init__(self, num_agents, embed_dim=32, channels=5, height=13, width=13):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        
        # CNN feature extractor for states
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),  # [batch_size*num_agents, 16, height, width]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [batch_size*num_agents, 16, height//2, width//2]
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),  # [batch_size*num_agents, 32, height//2, width//2]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [batch_size*num_agents, 32, height//4, width//4]
        )
        cnn_out_dim = (height // 4) * (width // 4) * channels  # Flattened output of CNN
        self.fc_state = nn.Linear(cnn_out_dim, 1)  # Reduce state feature to scalar per agent

        # Hyper-networks
        self.hyper_w1 = nn.Linear(1, num_agents * embed_dim)
        self.hyper_b1 = nn.Linear(1, embed_dim)
        self.hyper_w2 = nn.Linear(1, embed_dim)
        self.hyper_b2 = nn.Linear(1, 1)

    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)
        num_agents = agent_qs.size(1)
        states = states.view(batch_size * num_agents, states.size(2), states.size(3), states.size(4))
        
        # Process states with CNN
        states = self.cnn(states)  # [batch_size*num_agents, 32, height//4, width//4]
        states = states.reshape(batch_size * num_agents, -1)  # Flatten to [batch_size*num_agents, cnn_out_dim]
        states = self.fc_state(states)  # [batch_size*num_agents, 1]
        states = states.view(batch_size, num_agents, 1)  # Reshape to [batch_size, num_agents, 1]

        # Aggregate state to batch dimension
        global_state = states.mean(dim=1)  # Mean across agents: [batch_size, 1]

        # Get weights and biases from hyper-networks
        w1 = torch.abs(self.hyper_w1(global_state)).view(batch_size, num_agents, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(batch_size, 1, self.embed_dim)
        w2 = torch.abs(self.hyper_w2(global_state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)

        # Compute Mixing Network
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1  # [batch_size, 1, embed_dim]
        hidden = F.relu(hidden)
        q_total = torch.bmm(hidden, w2) + b2  # [batch_size, 1, 1]
        return q_total.squeeze(-1)  # [batch_size, 1]
