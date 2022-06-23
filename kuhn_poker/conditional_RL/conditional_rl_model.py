import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from collections import namedtuple
from itertools import count
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Normal, Categorical, OneHotCategorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR 
from embedding_learning.opponent_models import *

device = torch.device("cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim, action_dim, init_ort=False):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + latent_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)
        if init_ort:
            torch.nn.init.orthogonal_(self.fc1.weight)
            torch.nn.init.orthogonal_(self.fc3.weight)
            torch.nn.init.orthogonal_(self.fc4.weight)
            
    def forward(self, x, latent):
        x = torch.cat((x, latent), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim, init_ort=False):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + latent_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)
        if init_ort:
            torch.nn.init.orthogonal_(self.fc1.weight)
            torch.nn.init.orthogonal_(self.fc3.weight)
            torch.nn.init.orthogonal_(self.fc4.weight)

    def forward(self, x, latent):
        x = torch.cat((x, latent), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class PPO_VAE():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32
    value_loss_coef = 1.5
    entropy_loss_coef = 0.01

    def __init__(self, state_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, n_adv_pool, ckp_dir='../model_params/RL/'):
        super(PPO_VAE, self).__init__()
        self.actor_net = PolicyNet(state_dim, hidden_dim, embedding_dim, action_dim).to(device)
        self.critic_net = ValueNet(state_dim, hidden_dim, embedding_dim).to(device)
        self.n_adv_pool = n_adv_pool
        self.ckp_dir = ckp_dir

        # EncoderVAE 
        self.encoder = EncoderVAE(self.n_adv_pool, 128, embedding_dim).to(device)
        self.encoder.load_state_dict(torch.load(encoder_weight_path,map_location=torch.device('cpu')))
        print ('encoder weight loaded')

        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_parameters = list(self.actor_net.parameters())
        self.critic_parameters = list(self.critic_net.parameters())
        self.actor_optimizer = optim.Adam(self.actor_parameters, lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_parameters, lr=critic_lr)
        self.parameters = list(self.actor_net.parameters()) + list(self.critic_net.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=actor_lr)
            
    def select_action(self, state, latent):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state, latent)
        c = Categorical(action_prob)
        action = c.sample()
        u = np.zeros(2)
        u[action.item()] += 1
        return u, action.item(), action_prob[:,action.item()].item()
    
    def get_value(self, state, latent1, latent2, latent3):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            value = self.critic_net(state, latent1, latent2, latent3)
        return value.item()

    def evaluate(self, state, latent1, latent2, latent3, action):
        with torch.no_grad():
            action_prob = self.actor_net(state, latent1, latent2, latent3)
        dist = Categorical(action_prob)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def get_params(self):
        return {'actor': self.actor_net.state_dict(), 'critic': self.critic_net.state_dict()}
    
    def load_params(self, params):
        self.actor_net.load_state_dict(params['actor'])
        self.critic_net.load_state_dict(params['critic'])
    
    def save_params(self, ckp):
        save_dict = {'agent_params': self.get_params()}
        torch.save(save_dict, self.ckp_dir+'params_' + str(ckp) + '.pt', _use_new_zipfile_serialization=False)
    
    def init_from_save(self, filename):
        save_dict = torch.load(filename)
        self.load_params(save_dict['agent_params'])
        
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def update(self):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float).to(device)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.long).view(-1,1).to(device)
        latent = torch.tensor(np.array([t.latent for t in self.buffer]), dtype=torch.float).to(device)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1,1).to(device)

        returns = [t.returns for t in self.buffer]
        Gt = torch.tensor(returns, dtype=torch.float).to(device)        

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):


                Gt_index = Gt[index].view(-1,1)
                V = self.critic_net(state[index], latent[index])
                delta = Gt_index - V
                advantage = delta.detach()

                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index], latent[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage

                action_loss = -torch.min(surr1, surr2).mean() # Max->Min desent
                value_loss = F.mse_loss(Gt_index, V)

                # update actor network
                self.actor_optimizer.zero_grad()
                action_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                self.critic_optimizer.zero_grad()
                value_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.critic_parameters, self.max_grad_norm)
                self.critic_optimizer.step()
                
                self.training_step += 1

        del self.buffer[:] # clear experience
    


