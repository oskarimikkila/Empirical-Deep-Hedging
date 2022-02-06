"""
TD3 based on: https://arxiv.org/abs/1802.09477
"""
from include.settings import getSettings
from include.network import Actor, Critic
import torch
import torch.nn.functional as F
import include.utility as utility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(object):
    def __init__(self, state_dim, s = getSettings()):

        self.batch_size = s['batch_size']
        self.discount = s['discount']
        self.tau = s['tau']
        self.policy_noise = s['policy_noise']
        self.policy_noise_max = s['policy_noise_max']
        self.policy_freq = s['policy_freq']

        self.replay_buffer = utility.ReplayBuffer(state_dim, 1)
        self.total_it = 0

        self.actor = Actor(state_dim).to(device)
        self.actor_target = Actor(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=s['actor_lr'])

        self.critic1 = Critic(state_dim, 1, s).to(device)
        self.critic_target1 = Critic(state_dim, 1, s).to(device)
        self.critic_opt1 = torch.optim.Adam(self.critic1.parameters(), lr=s['critic_lr'])
        self.critic2 = Critic(state_dim, 1, s).to(device)
        self.critic_target2 = Critic(state_dim, 1, s).to(device)
        self.critic_opt2 = torch.optim.Adam(self.critic2.parameters(), lr=s['critic_lr'])

    def remember(self, cur_state, action, reward, new_state, done):
        self.replay_buffer.add(cur_state, action, new_state, reward, done)

    def forget(self):
        self.replay_buffer.empty()
        
    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self):
        if self.replay_buffer.size < self.batch_size + 1:
            return

        self.total_it += 1
        state, action, next_state, reward, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            noise = torch.rand_like(action) * self.policy_noise
            noise = noise.clamp(-self.policy_noise_max, self.policy_noise_max)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-1, 1)

            target_Q1 = self.critic_target1(next_state, next_action)
            target_Q2 = self.critic_target2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - dones) * self.discount * target_Q

        for Q, opt in zip([self.critic1, self.critic2], [self.critic_opt1, self.critic_opt2]):
            current_Q = Q(state, action)
            loss = F.mse_loss(current_Q, target_Q)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if self.total_it % self.policy_freq == 0:

            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # doesn't support further training
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))