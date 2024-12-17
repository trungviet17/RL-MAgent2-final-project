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


       
class QMix_Nets(nn.Module): 
    """"
    Mạng QMix học hàm Q-total -> với thông tin của state và các q-value của hàm trước đó.  
    Input: 
        1. Thông tin về state -> state_dim : kích thước state  
        2. Số lượng agent và hàm q max của agent đó  -> n_actions + số lượng agent 
    => setting 2 tham số tương ứng với 2 đầu vào  
    Output: 
        1. Giá trị Q-total 
    """

    def __init__(self, state_shape, n_actions, num_agent,  embed_dim = 32, hyper_embed_dim = 128): 
        super().__init__()

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.hyper_embed_dim = hyper_embed_dim
        
        # hyperparameter nhận đầu vào là kết quả từ từng GRU_QNets [num_agent, n_actions]
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_shape, hyper_embed_dim),
            nn.ReLU(),
            nn.Linear(hyper_embed_dim, num_agent * embed_dim)
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_shape, hyper_embed_dim),
            nn.ReLU(),
            nn.Linear(hyper_embed_dim, embed_dim)
        )

        self.hyper_b1 = nn.Linear(state_shape, embed_dim)
        self.hyper_b2 = nn.Linear(state_shape, 1)

        # dong vai tro la bias 
        self.V = nn.Sequential(
            nn.Linear(state_shape, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )



    def forward(self, agent_qs, state) :
        """
        Nhận đầu vào là một ma trận 
        
        Input: 


        Output: 


        
        """
        batch_size = state.size(0)

        w1 = self.hyper_w1(state).view(batch_size, -1, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        w2 = self.hyper_w2(state).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        hidden = torch.bmm(agent_qs, w1) + b1
        hidden = F.relu(hidden)

        q_total = torch.bmm(hidden, w2) + b2
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


