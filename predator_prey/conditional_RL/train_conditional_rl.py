import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F

from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *
import multiagent.scenarios as scenarios
from embedding_learning.opponent_models import *
from embedding_learning.data_generation import get_all_adv_policies

from conditional_RL.conditional_rl_model import PPO_VAE
from utils.multiple_test import play_multiple_times_train

N_ADV = 3
adv_pool = ['PolicyN', 'PolicyEA', 'PolicyW', 'PolicyA']
n_adv_pool = len(adv_pool)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_three_adv_all_policies(env, adv_pool, adv_path):
    all_policies_idx0 = get_all_adv_policies(env,adv_pool,adv_path,agent_index=0)
    all_policies_idx1 = get_all_adv_policies(env,adv_pool,adv_path,agent_index=1)
    all_policies_idx2 = get_all_adv_policies(env,adv_pool,adv_path,agent_index=2)
    all_policies = [all_policies_idx0,all_policies_idx1,all_policies_idx2]
    return all_policies

def main(args):
    gamma = 0.99
    Transition_vae = namedtuple('Transition_vae', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'latent'])
    
    hidden_dim = 128
    seed = 1
    actor_lr = args.lr1
    critic_lr = args.lr2

    num_episodes = args.num_episodes
    checkpoint_freq = 1000
    adv_change_freq = 1
    use_latent = True
    use_mean = False
    batch_size = 1000
    ppo_update_freq = 10
    test_freq = 100
    if not use_latent:
        use_mean = False
    
    exp_id = args.version
    settings = {}
    settings['exp_id'] = exp_id
    settings['hidden_dim'] = hidden_dim
    settings['actor_lr'] = actor_lr
    settings['critic_lr'] = critic_lr
    settings['use_latent'] = use_latent
    settings['use_mean'] = use_mean
    settings['batch_size'] = batch_size
    settings['ppo_update_freq'] = ppo_update_freq
    settings['adv_change_freq'] = adv_change_freq
    settings['seed'] = seed

    print(settings)
    scenario = scenarios.load(args.scenario).Scenario()
    world_vae = scenario.make_world()
    
    env_vae = MultiAgentEnv(world_vae, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None, shared_viewer=False, discrete_action=True)
    
    env_vae.seed(seed)
    
    np.random.seed(seed)
    state_dim = env_vae.observation_space[env_vae.n-1].shape[0]
    action_dim = env_vae.action_space[env_vae.n-1].n

    embedding_dim = 2
    if not use_latent:
        embedding_dim = n_adv_pool
    
    encoder_weight_path = args.encoder_file

    agent_vae = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, gamma, n_adv_pool)
    agent_vae.batch_size = batch_size
    agent_vae.ppo_update_time = ppo_update_freq

    return_list_vae = []
    test_list_vae = []

    adv_path = [None] * len(adv_pool)
    all_policies_vae = get_three_adv_all_policies(env_vae, adv_pool, adv_path)
    
    selected_adv_idx = 0
    for i in range(50):
        with tqdm(total=int(num_episodes/50), desc="Iteration %d" %i) as pbar:
            for i_episode in range(int(num_episodes/50)):
                if i_episode % adv_change_freq == 0:
                    policies_vae = []
                    policy_vec = np.zeros(n_adv_pool)
                    selected_adv_idx = np.random.randint(0,4)
                    policy_vec[selected_adv_idx] += 1
                    for j in range(N_ADV):
                        policies_vae.append(all_policies_vae[j][selected_adv_idx])
                
                episode_return_vae = 0
                obs_n_vae = env_vae._reset()
                policy_vec_tensor = torch.tensor([policy_vec], dtype=torch.float).to(device)

                for st in range(args.steps):
                    act_n_vae = []

                    # vae
                    for j, policy in enumerate(policies_vae):
                        act_vae = policy.action(obs_n_vae[j])
                        act_n_vae.append(act_vae)
                    if use_latent:
                        latent, mu, _ = agent_vae.encoder(policy_vec_tensor)
                    else:
                        latent = policy_vec_tensor.clone().detach()
                    if use_mean:
                        latent = mu
                    act_vae, act_index_vae, act_prob_vae = agent_vae.select_action(obs_n_vae[3], latent,2)
                    act_n_vae.append(act_vae)
                    next_obs_n_vae, reward_n_vae, _,_ = env_vae._step(act_n_vae)
                    latent = latent[0].cpu().detach().numpy()

                    trans_vae = Transition_vae(obs_n_vae[3], act_index_vae, act_prob_vae, reward_n_vae[3], next_obs_n_vae[3], latent)
                    agent_vae.store_transition(trans_vae)
                    episode_return_vae += reward_n_vae[3]
                    obs_n_vae = next_obs_n_vae
                
                if len(agent_vae.buffer) >= agent_vae.batch_size:
                    agent_vae.update(i_episode)
                    
                return_list_vae.append(episode_return_vae)
                
                if i_episode % 50 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/50 * i + i_episode),
                                        'return_vae': '%.3f' % np.mean(return_list_vae[-50:])})
                pbar.update(1)
                current_episode = num_episodes / 50 * i + i_episode + 1
                if current_episode % checkpoint_freq == 0:
                    agent_vae.save_params(exp_id + '_' + str(current_episode))
                
                if current_episode % test_freq == 0:
                    play_episodes = 100
                    test_list_vae = test_list_vae + play_multiple_times_train(env_vae, agent_vae, all_policies_vae, 'vae', 'rule', play_episodes=play_episodes)

                    result_dict = {}
                    result_dict['version'] = exp_id
                    result_dict['num_episodes'] = len(return_list_vae)
                    result_dict['return_list_vae'] = return_list_vae
                    result_dict['test_list_vae'] = test_list_vae
                    result_dict['settings'] = settings

                    pickle.dump(result_dict, open('results/return_' + exp_id + '.p', "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l1', '--lr1', default=1e-4, help='Actor learning rate')
    parser.add_argument('-l2', '--lr2', default=1e-4, help='Critic learning rate')
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, help='Num of steps in a single run')
    parser.add_argument('-ep', '--num_episodes', default=10000, help='Num of episodes')
    parser.add_argument('-v', '--version', default='v0')
    parser.add_argument('-e', '--encoder_file', default='../model_params/VAE/encoder_vae_param_demo.pt', help='file name of the encoder parameters')
    args = parser.parse_args()

    main(args)