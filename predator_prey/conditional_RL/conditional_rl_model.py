import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from embedding_learning.opponent_models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.lstm = torch.nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)
            
    def forward(self, obs, action, hidden, latent):
        x = torch.cat((obs, action), dim=-1)
        h, hidden = self.lstm(x, hidden)
        h = h[:,-1,:]
        x = F.relu(self.fc1(h))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, latent), -1)
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim):
        super(ValueNet, self).__init__()
        self.lstm = torch.nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, action, hidden, latent):
        x = torch.cat((obs, action), dim=-1)
        h, hidden = self.lstm(x, hidden)
        h = h[:,-1,:]
        x = F.relu(self.fc1(h))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, latent), -1)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class PPO_VAE():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 20
    buffer_capacity = 1000
    batch_size = 1000

    def __init__(self, state_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, gamma, n_adv_pool=4):
        super(PPO_VAE, self).__init__()

        self.hidden_dim = hidden_dim 
        self.actor_net = PolicyNet(state_dim, self.hidden_dim, embedding_dim, action_dim).to(device)
        self.critic_net = ValueNet(state_dim, self.hidden_dim, embedding_dim).to(device)

        # EncoderVAE
        self.encoder = EncoderVAE(n_adv_pool, hidden_dim, embedding_dim).to(device)
        print("encoder weight loaded")
        self.encoder.load_state_dict(torch.load(encoder_weight_path))

        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.gamma = gamma

            
    def select_action(self, obs, act, hidden, latent, dim_c):
        with torch.no_grad():
            action_prob = self.actor_net(obs, act, hidden, latent)
        c = Categorical(action_prob)
        action = c.sample()
        u = np.zeros(5)
        u[action.item()] += 1
        return np.concatenate([u, np.zeros(dim_c)]), action.item(), action_prob[:,action.item()].item()

    def get_params(self):
        return {'actor': self.actor_net.state_dict(), 'critic': self.critic_net.state_dict()}
    
    def load_params(self, params):
        self.actor_net.load_state_dict(params['actor'])
        self.critic_net.load_state_dict(params['critic'])
    
    def save_params(self, ckp):
        save_dict = {'agent_params': self.get_params()}
        torch.save(save_dict, '../model_params/RL/params_' + str(ckp) + '.pt', _use_new_zipfile_serialization=False)
    
    def save_params_with_path(self, path):
        save_dict = {'agent_params': self.get_params()}
        torch.save(save_dict, path, _use_new_zipfile_serialization=False)
        
    def init_from_save(self, filename):
        save_dict = torch.load(filename)
        self.load_params(save_dict['agent_params'])
        
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
    
    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1,1).to(device)
        latent = torch.tensor([t.latent for t in self.buffer], dtype=torch.float).to(device)

        obs_traj = torch.tensor([t.obs_traj for t in self.buffer], dtype=torch.float).to(device)
        act_traj = torch.tensor([t.act_traj for t in self.buffer], dtype=torch.float).to(device)

        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1,1).to(device)

        R = 0
        Gt = []
        ct = 0
        for r in reward[::-1]:
            R = r + self.gamma * R
            ct += 1
            Gt.insert(0, R)
            if ct >= 50:
                R = 0
                ct = 0
        
        Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):

                hidden = [torch.zeros((1, len(index), self.hidden_dim)).to(device), torch.zeros((1, len(index), self.hidden_dim)).to(device)]

                Gt_index = Gt[index].view(-1,1)
                V = self.critic_net(obs_traj[index],  act_traj[index], hidden, latent[index])
                delta = Gt_index - V
                advantage = delta.detach()

                action_prob = self.actor_net(obs_traj[index],  act_traj[index], hidden, latent[index]).gather(1, action[index])

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean() # Max->Min desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.training_step += 1

        del self.buffer[:] # clear experience
