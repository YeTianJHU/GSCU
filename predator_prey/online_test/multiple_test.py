import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import torch

from multiagent.mypolicy import *
from VAE.opponent_models import *

N_ADV = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# used for ppo, agentb, and vae
# adv_type = {'rule', 'rl'}, agent_type = {'rule', 'ppo', 'vae'}
def play_multiple_times_without_update_simple(env, agent, adv1, adv2, adv3, agent_type, adv_type, play_episodes=10, latent=None):
    return_list = []
    for i in range(play_episodes):
        obs_n = env._reset()
        episode_return = 0
        for st in range(50):
            act_n = []
            if adv_type == 'rule':
                act1 = adv1.action(obs_n[0])
                act2 = adv2.action(obs_n[1])
                act3 = adv3.action(obs_n[2])
            else:
                act1,_,_ = adv1.select_action(obs_n[0], 2)
                act2,_,_ = adv2.select_action(obs_n[1], 2)
                act3,_,_ = adv3.select_action(obs_n[2], 2)
            act_n.append(act1)
            act_n.append(act2)
            act_n.append(act3)
            
            if agent_type == 'rule':
                act = agent.action(obs_n[3])
            elif agent_type == 'ppo':
                act,_,_ = agent.select_action(obs_n[3], 2)
            else:
                if latent is None:
                    print("Please input latent to use VAE")
                act,_,_ = agent.select_action(obs_n[3], latent, 2)
            act_n.append(act)

            next_obs_n, reward_n, done_n, _ = env._step(act_n)
            episode_return += reward_n[env.n-1]
            obs_n = next_obs_n
        return_list.append(episode_return)
    return return_list

# used for bandit, which is composed of vae and agentb
# adv_type = {'rule', 'rl'}, agent_type = {'bandit'}
def play_multiple_times_without_update_bandit(env, agent_agb, agent_VAE, adv1, adv2, adv3, adv_type, play_episodes=10,
                                            latent=None, p=0, use_exp3=False):
    return_list = []
    for i in range(play_episodes):
        obs_n = env._reset()
        episode_return = 0
        random_select = np.random.random()
        if not use_exp3:
            random_select = 1
        for st in range(50):
            act_n = []
            if adv_type == 'rule':
                act1 = adv1.action(obs_n[0])
                act2 = adv2.action(obs_n[1])
                act3 = adv3.action(obs_n[2])
            else:
                act1,_,_ = adv1.select_action(obs_n[0], 2)
                act2,_,_ = adv2.select_action(obs_n[1], 2)
                act3,_,_ = adv3.select_action(obs_n[2], 2)
            act_n.append(act1)
            act_n.append(act2)
            act_n.append(act3)
            
            if random_select > p:
                act = agent_agb.action(obs_n[3])
            else:
                act,_,_ = agent_VAE.select_action(obs_n[3], latent, 2)
            act_n.append(act)

            next_obs_n, reward_n, done_n, _ = env._step(act_n)
            episode_return += reward_n[env.n-1]
            obs_n = next_obs_n
        return_list.append(episode_return)
    return return_list