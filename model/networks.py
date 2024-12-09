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


class LSTM_QNets(nn.Module):
    def __init__(self, n_observation, n_actions, hidden_dim: int = 120):
        super(LSTM_QNets, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_observation[-1], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.rnn = nn.LSTM(input_size=64 * n_observation[0] * n_observation[1], hidden_size=hidden_dim, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x, hidden_state):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        
        rnn_out, hidden_state = self.rnn(cnn_out, hidden_state)
        
        q_values = self.fc(rnn_out)
        
        return q_values, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.rnn.hidden_size).to(next(self.parameters()).device),
                torch.zeros(1, batch_size, self.rnn.hidden_size).to(next(self.parameters()).device))
    

class GRU_QNets(nn.Module): 
    """
    Mạng GRU học ra hàm Qa cho thuật toán QMix (hàm của mỗi agent con)
    Input: 
        1. n_observation : tuple - kích thước của ảnh đầu vào
        2. n_actions : so luong hanh dong thuc hien 
    Output: 
        1. Hàm Q của 1 agent 
    """
    def __init__(self, n_observation, n_actions, hidden_dim: int = 120):
        super(GRU_QNets, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(n_observation, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
        )
        self.gru = nn.GRU(input_size=hidden_dim , hidden_size=hidden_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x, hidden_state = None ):

        if hidden_state is None:
            outputs, hidden_state = self.gru(x)
        else:
            outputs, hidden_state = self.gru(x, hidden_state)

        outputs = outputs[:, -1, :]

        q_values = self.fc(outputs)

        return q_values, hidden_state
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.gru.hidden_size).to(next(self.parameters()).device)

       
class QMix_Nets(nn.Module): 
    """"
    Mạng QMix học hàm Q-total -> với thông tin của state và các q-value của hàm trước đó.  
    Input: 
        1. state_shape : int - kích thước của state
        2. n_actions : int - số lượng hành động



    """

    def __init__(self, state_shape, n_actions, num_agent,  embed_dim = 32, hyper_embed_dim = 128): 
        super().__init__()

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.hyper_embed_dim = hyper_embed_dim

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
        Nhận đầu vào là một chuỗi 
        
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

