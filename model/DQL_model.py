import torch.nn as nn
import torch

class DRQNets(nn.Module):
    def __init__(self, n_observation, n_actions, hidden_dim: int = 120):
        super(DRQNets, self).__init__()
        
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