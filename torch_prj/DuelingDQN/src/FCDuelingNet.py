import torch.nn as nn
import torch.nn.functional as F

class FCNDueling(nn.Module):
  def __init__(self, n_input, n_output, n_hidden=(32, 32), activation_fc=F.relu):
    super(FCNDueling, self).__init__()
    self.input_layer = nn.Linear(n_input, n_hidden[0])
    self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden[i], n_hidden[i + 1]) for i in range(len(n_hidden) - 1)])
    self.value_output = nn.Linear(n_hidden[-1], 1)# instead of output layer we use <- value output
    self.advandage_output = nn.Linear(n_hidden[-1], n_output)# and <- advandage output (q = v + a)
    self.activation_fc = activation_fc
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = self.activation_fc(self.input_layer(x))
    x = self.dropout(x)
    for hidden_layer in self.hidden_layers:
      x = self.activation_fc(hidden_layer(x))

    # that's all difference
    a = self.advandage_output(x)
    v = self.value_output(x)
    v = v.expand_as(a)

    q = v + a - a.mean((0), keepdim=True).expand_as(a)

    return q