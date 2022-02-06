from include.settings import getSettings

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, settings = getSettings()):
        super(Actor, self).__init__()
        
        self.lrelu_alpha = settings['lrelu_alpha']
        actor_nn = settings['actor_nn']

        self.input = nn.Linear(state_dim, actor_nn)
        self.hidden = nn.Linear(actor_nn, actor_nn)
        self.output = nn.Linear(actor_nn, 1)

    def forward(self, state):
        x = F.leaky_relu(self.input(state), self.lrelu_alpha)
        x = F.leaky_relu(self.hidden(x), self.lrelu_alpha)
        x = torch.tanh(self.output(x))

        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, settings = getSettings()):
        super(Critic, self).__init__()
        
        self.lrelu_alpha = settings['lrelu_alpha']
        critic_nn = settings['critic_nn']
                
        self.input = nn.Linear(state_dim + action_dim, critic_nn)
        self.hidden1 = nn.Linear(critic_nn, critic_nn)
        self.hidden2 = nn.Linear(critic_nn, critic_nn)
        self.output = nn.Linear(critic_nn, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.leaky_relu(self.input(x), self.lrelu_alpha)
        x = F.leaky_relu(self.hidden1(x), self.lrelu_alpha)
        x = F.leaky_relu(self.hidden2(x), self.lrelu_alpha)
        x = self.output(x)
        
        return x