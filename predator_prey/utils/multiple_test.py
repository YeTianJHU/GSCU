import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import torch
import torch.nn.functional as F
from utils.config_predator_prey import Config

N_ADV = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# used for ppo, rule, and vae agent
# used for validation in rl_training phrase
# adv_type = {'rule', 'rl'}, agent_type = {'rule', 'ppo', 'vae'}
def play_multiple_times_train(env, agent, adv_policies, agent_type, adv_type, play_episodes=10):
    return_list = []
    n_adv = len(adv_policies[0])
    window_size = Config.WINDOW_SIZW
    state_dim = env.observation_space[env.n-1].shape[0]
    modelled_act_dim = env.action_space[env.n-1].n
    hidden_dim = Config.HIDDEN_DIM

    for i in range(play_episodes):
        obs_n = env._reset()
        episode_return = 0
        selected_adv_idx = np.random.randint(0,n_adv)
        adv1 = adv_policies[0][selected_adv_idx]
        adv2 = adv_policies[1][selected_adv_idx]
        adv3 = adv_policies[2][selected_adv_idx]
        policy_vec = np.zeros(n_adv)
        policy_vec[selected_adv_idx] += 1
        policy_vec_tensor = torch.tensor([policy_vec], dtype=torch.float).to(device)

        obs_traj = [np.zeros(state_dim)]*(window_size-1)
        act_traj = [np.zeros(modelled_act_dim)]*window_size
        hidden = [torch.zeros((1,1,hidden_dim)).to(device), torch.zeros((1,1,hidden_dim)).to(device)]

        if agent_type == 'vae':
            _,latent,_ = agent.encoder(policy_vec_tensor)
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
                    print("Please enter latent to use VAE")
                obs_traj.append(obs_n[3])
                obs_traj_tensor = torch.tensor([obs_traj], dtype=torch.float).to(device)
                act_traj_tensor = torch.tensor([act_traj], dtype=torch.float).to(device)
                act,_,_ = agent.select_action(obs_traj_tensor, act_traj_tensor, hidden, latent, 2)
            act_n.append(act)

            next_obs_n, reward_n, _,_ = env._step(act_n)
            episode_return += reward_n[3]
            obs_n = next_obs_n

            if len(obs_traj) >= window_size:
                obs_traj.pop(0)
                act_traj.pop(0)
            act_traj.append(act[:-2]) # use action dim as 5

        return_list.append(episode_return)
    return return_list

# used for ppo, rule, and vae agent
# used for testing in online_test phrase
# adv_type = {'rule', 'rl'}, agent_type = {'rule', 'ppo', 'vae'}
def play_multiple_times_test(env, agent, adv1, adv2, adv3, agent_type, adv_type, play_episodes=10, latent=None):
    return_list = []
    window_size = Config.WINDOW_SIZW
    state_dim = env.observation_space[env.n-1].shape[0]
    modelled_act_dim = env.action_space[env.n-1].n
    hidden_dim = Config.HIDDEN_DIM

    for i in range(play_episodes):
        obs_n = env._reset()
        episode_return = 0

        obs_traj = [np.zeros(state_dim)]*(window_size-1)
        act_traj = [np.zeros(modelled_act_dim)]*window_size
        hidden = [torch.zeros((1,1,hidden_dim)).to(device), torch.zeros((1,1,hidden_dim)).to(device)]

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
                obs_traj.append(obs_n[3])
                obs_traj_tensor = torch.tensor([obs_traj], dtype=torch.float).to(device)
                act_traj_tensor = torch.tensor([act_traj], dtype=torch.float).to(device)
                act,_,_ = agent.select_action(obs_traj_tensor, act_traj_tensor, hidden, latent, 2)
            act_n.append(act)

            next_obs_n, reward_n, done_n, _ = env._step(act_n)
            episode_return += reward_n[env.n-1]
            obs_n = next_obs_n

            if len(obs_traj) >= window_size:
                obs_traj.pop(0)
                act_traj.pop(0)
            act_traj.append(act[:-2]) # use action dim as 5

        return_list.append(episode_return)
    return return_list

# used for bandit, which is composed of vae and rule agent
# adv_type = {'rule', 'rl'}, agent_type = {'bandit'}
def play_multiple_times_test_bandit(env, agent_pi, agent_VAE, adv1, adv2, adv3, adv_type, play_episodes=10,
                                            latent=None, p=0, use_exp3=False):
    return_list = []
    window_size = Config.WINDOW_SIZW
    state_dim = env.observation_space[env.n-1].shape[0]
    modelled_act_dim = env.action_space[env.n-1].n
    hidden_dim = Config.HIDDEN_DIM

    for i in range(play_episodes):
        obs_n = env._reset()
        episode_return = 0

        obs_traj = [np.zeros(state_dim)]*(window_size-1)
        act_traj = [np.zeros(modelled_act_dim)]*window_size
        hidden = [torch.zeros((1,1,hidden_dim)).to(device), torch.zeros((1,1,hidden_dim)).to(device)]

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
                act,_,_ = agent_pi.select_action(obs_n[3],2)
            else:
                obs_traj.append(obs_n[3])
                obs_traj_tensor = torch.tensor([obs_traj], dtype=torch.float).to(device)
                act_traj_tensor = torch.tensor([act_traj], dtype=torch.float).to(device)
                act,_,_ = agent_VAE.select_action(obs_traj_tensor, act_traj_tensor, hidden, latent, 2)
            act_n.append(act)

            next_obs_n, reward_n, done_n, _ = env._step(act_n)
            episode_return += reward_n[env.n-1]
            obs_n = next_obs_n

            if len(obs_traj) >= window_size:
                obs_traj.pop(0)
                act_traj.pop(0)
            act_traj.append(act[:-2]) # use action dim as 5

        return_list.append(episode_return)
    return return_list