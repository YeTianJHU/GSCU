import gym
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from collections import namedtuple

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import time

from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *

import multiagent.scenarios as scenarios
from VAE.opponent_models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)
            
    def forward(self, x, latent):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, latent), -1)
        x = F.relu(self.fc3(x))

        return F.softmax(self.fc4(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x, latent):
        x = F.relu(self.fc1(x))
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
        self.actor_net = PolicyNet(state_dim, hidden_dim, embedding_dim, action_dim).to(device)
        self.critic_net = ValueNet(state_dim, hidden_dim, embedding_dim).to(device)

        # EncoderVAE
        self.encoder = EncoderVAE(n_adv_pool, hidden_dim, embedding_dim).to(device)
        if embedding_dim == 2 or embedding_dim == 8:
            print("encoder weight loaded")
            self.encoder.load_state_dict(torch.load(encoder_weight_path))

        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.gamma = gamma

            
    def select_action(self, state, latent, dim_c):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state, latent)

        c = Categorical(action_prob)
        action = c.sample()
        u = np.zeros(5)
        u[action.item()] += 1
        return np.concatenate([u, np.zeros(dim_c)]), action.item(), action_prob[:,action.item()].item()
    
    def get_value(self, state, latent):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            value = self.critic_net(state, latent)
        return value.item()

    def get_params(self):
        return {'actor': self.actor_net.state_dict(), 'critic': self.critic_net.state_dict()}
    
    def load_params(self, params):
        self.actor_net.load_state_dict(params['actor'])
        self.critic_net.load_state_dict(params['critic'])
    
    def save_params(self, ckp):
        save_dict = {'agent_params': self.get_params()}
        torch.save(save_dict, 'trained_parameters/GSCU_Greedy/params_' + str(ckp) + '.pt', _use_new_zipfile_serialization=False)
    
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
                Gt_index = Gt[index].view(-1,1)
                V = self.critic_net(state[index], latent[index])
                delta = Gt_index - V
                advantage = delta.detach()

                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index], latent[index]).gather(1, action[index]) # new policy

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

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2) 
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l1', '--lr1', default=1e-4, help='Actor learning rate')
    parser.add_argument('-l2', '--lr2', default=1e-4, help='Critic learning rate')
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-ep', '--num_episodes', default=3000, help='Num of episodes')
    args = parser.parse_args()
    
    gamma = 0.99
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'latent'])
    hidden_dim = 128
    seed = 1
    torch.manual_seed(seed)
    actor_lr = args.lr1
    critic_lr = args.lr2
    num_episodes = args.num_episodes
    scenario = scenarios.load(args.scenario).Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        info_callback=None, shared_viewer=False, discrete_action=True)
    env.seed(seed)
    env._render()

    state_dim = env.observation_space[env.n-1].shape[0]
    action_dim = env.action_space[env.n-1].n

    # Newly added
    embedding_dim = 2
    encoder_weight_path = '../VAE/saved_model_params/encoder_vae_param.pt'
    agent = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, gamma)

    # Need to be modified
    policies = [PolicyA(env,i) for i in range(env.n-1)]
    adv_index = torch.tensor([0]).to(device)
    onehot_vec = F.one_hot(adv_index, num_classes=4).float().to(device)
    
    return_list = []
    for i in range(50):
        with tqdm(total=int(num_episodes/50), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/50)):
                episode_return = 0
                obs_n = env._reset()
                done = False
                ct = 0

                for st in range(50):
                    # Newly added for embedding
                    latent, _, _ = agent.encoder(onehot_vec)
                    
                    act, act_index, action_prob = agent.select_action(obs_n[env.n-1], latent, env.world.dim_c)
            
                    act_n = [] # env action
                    for j, policy in enumerate(policies):
                        act_n.append(policy.action(obs_n[j]))
                    act_n.append(act)

                    next_obs_n, reward_n, done_n, _ = env._step(act_n)

                    latent = latent[0].cpu().detach().numpy()
                    
                    trans = Transition(obs_n[env.n-1], act_index, action_prob, reward_n[env.n-1], next_obs_n[env.n-1], latent)
                    agent.store_transition(trans)
                    episode_return += reward_n[env.n-1]
                    
                    obs_n = next_obs_n
                    
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_episode)
                    
                return_list.append(episode_return)
                if (i_episode+1) % 50 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/50 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-50:])})
                pbar.update(1)
        

    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, 49)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO_VAE')
    plt.savefig("PPO_VAE.jpg")



