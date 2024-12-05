#src/models.py
import torch
import torch.nn as nn
from typing import Dict

class LSTMModel(nn.Module):
    def __init__(self, config: Dict):
        super(LSTMModel, self).__init__()
        self.input_dim = len(config['weather_features'])
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output
        last_output = lstm_out[:, -1, :]
        
        # Predict
        out = self.fc(last_output)
        return out