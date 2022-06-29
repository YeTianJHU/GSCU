import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Normal, Categorical, OneHotCategorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from collections import namedtuple
from itertools import count
import argparse
import time

device = torch.device("cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self,state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    batch_size = 1024

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr):
        super(PPO, self).__init__()
        self.actor_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_net = ValueNet(state_dim, hidden_dim).to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)

    def select_action(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        u = np.zeros(2)
        u[action.item()] += 1
        return u, action.item(), action_prob[:,action.item()].item()
    
    def get_value(self, state):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()
    
    def get_params(self):
        return {'actor': self.actor_net.state_dict(), 'critic': self.critic_net.state_dict()}
    
    def load_params(self, params):
        self.actor_net.load_state_dict(params['actor'])
        self.critic_net.load_state_dict(params['critic'])
    
    def save_params(self, ckp):
        save_dict = {'agent_params': self.get_params()}
        torch.save(save_dict, 'trained_parameters/PPO/params_' + str(ckp) + '.pt', _use_new_zipfile_serialization=False)
    
    def init_from_save(self, filename):
        save_dict = torch.load(filename)
        self.load_params(save_dict['agent_params'])

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
    
    def update(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1,1).to(device)
        returns = [t.returns for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1,1).to(device)
        Gt = torch.tensor(returns, dtype=torch.float).to(device)
        
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                Gt_index = Gt[index].view(-1,1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()

                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean() # Max->Min desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.training_step += 1

        del self.buffer[:] # clear experience


