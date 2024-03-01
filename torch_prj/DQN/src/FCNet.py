import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
  def __init__(self, n_input, n_output, n_hidden=(32, 32), activation_fc=F.relu):
    super(FCN, self).__init__()
    self.input_layer = nn.Linear(n_input, n_hidden[0])
    self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden[i], n_hidden[i + 1]) for i in range(len(n_hidden) - 1)])
    self.output_layer = nn.Linear(n_hidden[-1], n_output)
    self.activation_fc = activation_fc
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = self.activation_fc(self.input_layer(x))
    x = self.dropout(x)
    for hidden_layer in self.hidden_layers:
      x = self.activation_fc(hidden_layer(x))
    x = self.output_layer(x)

    return x