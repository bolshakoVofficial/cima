import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_agents, n_actions, name, checkpoint_dir):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name, checkpoint_dir):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.pi = nn.Linear(64, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class AutoEncoderLinear(nn.Module):
    def __init__(self, input_dims, n_actions, name, lr, checkpoint_dir):
        super(AutoEncoderLinear, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, name)

        self.encoder = nn.Sequential(
            nn.Linear(input_dims + n_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dims),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
