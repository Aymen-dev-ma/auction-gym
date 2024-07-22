# PolicyNetwork.py
import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        action = np.random.choice(len(probs), p=probs.detach().numpy())
        return action, probs[action]
