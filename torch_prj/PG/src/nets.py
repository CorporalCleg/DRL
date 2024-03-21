import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class FCN(nn.Module):
  def __init__(self, n_input, n_output, n_hidden=(32, 32), activation_fc=F.relu, device='cpu'):
    super(FCN, self).__init__()
    self.input_layer = nn.Linear(n_input, n_hidden[0])
    self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden[i], n_hidden[i + 1]) for i in range(len(n_hidden) - 1)])
    self.output_layer = nn.Linear(n_hidden[-1], n_output)
    self.activation_fc = activation_fc
    self.dropout = nn.Dropout(0.2)
    self.device = device

  def forward(self, x):
    x = self._format(x)
    x = self.activation_fc(self.input_layer(x))
    x = self.dropout(x)
    for hidden_layer in self.hidden_layers:
      x = self.activation_fc(hidden_layer(x))
    x = self.output_layer(x)

    return x
  
  def _format(self, state):
    x = state
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)
    
    return x.to(self.device)

  def full_pass(self, state):
    logits = self.forward(state)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    logpa = dist.log_prob(action).unsqueeze(-1)
    entropy = dist.entropy().unsqueeze(-1)
    is_exploratory = action.detach().cpu().numpy() != np.argmax(logits.detach().cpu().numpy())

    return action.item(), logpa, is_exploratory.item(), entropy
  
  def select_action(self, state):
    with torch.no_grad():  
      logits = self.forward(state)
      dist = torch.distributions.Categorical(logits=logits)
      action = dist.sample()
      return action.item()
    

class ConvNet(nn.Module):
  def __init__(self, n_input, n_output, n_hidden=64, activation_fc=F.relu, kernel_size=5, n_kernels=4, device='cpu'):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv1d(1, n_kernels, kernel_size)
    self.linear0 = nn.Linear((n_input - kernel_size + 1) * n_kernels, n_hidden * 2)
    self.linear1 = nn.Linear(n_hidden * 2, n_hidden)
    self.output_layer = nn.Linear(n_hidden, n_output)
    self.activation_fc = activation_fc
    self.dropout = nn.Dropout(0.2)
    self.device = device

  def forward(self, state):
    x = self._format(state)
    x = self.activation_fc(self.conv1(x))
    x = self.dropout(torch.flatten(x))
    x = self.activation_fc(self.linear0(x))
    x = self.activation_fc(self.linear1(x))
    x = self.output_layer(x)

    return x
  
  def _format(self, state):
    x = state
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)
    
    return x.to(self.device)

  def full_pass(self, state):
    logits = self.forward(state)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    logpa = dist.log_prob(action).unsqueeze(-1)
    entropy = dist.entropy().unsqueeze(-1)
    is_exploratory = action.detach().cpu().numpy() != np.argmax(logits.detach().cpu().numpy())

    return action.item(), logpa, is_exploratory, entropy
  
  def select_action(self, state):
    with torch.no_grad():  
      logits = self.forward(state)
      dist = torch.distributions.Categorical(logits=logits)
      action = dist.sample()
      return action.item()
    