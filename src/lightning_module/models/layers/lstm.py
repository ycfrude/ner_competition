# author: 
# contact: ycfrude@163.com
# datetime:2022/4/1 7:51 PM
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout_prob=0.1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, num_layers=num_layers, batch_first=batch_first)
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input):
        output, (hn, cn) = self.lstm(input)
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output

