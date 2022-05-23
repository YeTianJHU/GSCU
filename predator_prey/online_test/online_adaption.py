import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
from collections import namedtuple
import pickle
import torch

from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *
import multiagent.scenarios as scenarios
from VAE.opponent_models import OpponentModel
from VAE.data_generation import get_all_adv_policies
from VAE.bayesian_update import BayesianUpdater, VariationalInference, EXP3

from conditional_RL.GSCU_Greedy import PPO_VAE
from online_test.multiple_test import *

N_ADV = 3
seen_adv_pool = ['PolicyN', 'PolicyEA', 'PolicyW', 'PolicyA']
unseen_adv_pool = ['PolicyStay', 'PolicyS', 'PolicyRL1', 'PolicyRL2']
mix_adv_pool = seen_adv_pool + unseen_adv_pool
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_three_adv_all_policies(env, adv_pool, adv_path):
    all_policies_idx0 = get_all_adv_policies(env,adv_pool,adv_path,agent_index=0)
    all_policies_idx1 = get_all_adv_policies(env,adv_pool,adv_path,agent_index=1)
    all_policies_idx2 = get_all_adv_policies(env,adv_pool,adv_path,agent_index=2)
    all_policies = [all_policies_idx0,all_policies_idx1,all_policies_idx2]
    return all_policies

def main(args):
    dataloader = open('policy_vec_sequence_4.p', 'rb')
    data = pickle.load(dataloader)
    policy_vec_seq = data['policy_vec_seq']

    gamma = 0.99
    hidden_dim = 128
    seed = 1
    actor_lr = args.lr1
    critic_lr = args.lr2
    num_episodes = args.num_episodes
    adv_change_freq = 200
    adv_pool_type = 'seen' # option: 'seen', 'unseen', 'mix'

    ckp_num = 10000
    ckp_freq = 20
    exp_num = 4
    test_num = 14
    test_id = 'v' + str(exp_num) + '_' + adv_pool_type + '_' + str(test_num)
    print(test_id)

    scenario = scenarios.load(args.scenario).Scenario()
    world_vae = scenario.make_world()
    world_bandit = scenario.make_world()
    

    env_vae = MultiAgentEnv(world_vae, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None, shared_viewer=False, discrete_action=True)
    env_bandit = MultiAgentEnv(world_bandit, scenario.reset_world, scenario.reward, scenario.observation,
                                info_callback=None, shared_viewer=False, discrete_action=True)
    env_vae.seed(seed)
    env_bandit.seed(seed)

    state_dim = env_vae.observation_space[3].shape[0]
    action_dim = env_vae.action_space[3].n
    embedding_dim = 2
    encoder_weight_path = '../VAE/saved_model_params/encoder_vae_param.pt'
    decoder_weight_path = '../VAE/saved_model_params/decoder_param.pt'

    
    agent_vae = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, gamma, 4)
    agent_bandit_pi = ConservativePolicy(env_bandit, env_bandit.n-1, epsilon=0.2)
    

    ppo_vae_path = '../conditional_RL/trained_parameters/GSCU_Greedy/params_' + 'v57' + '_' + str(ckp_num) + '.0.pt'
    agent_vae.init_from_save(ppo_vae_path)

    return_list_vae = []
    return_list_bandit = []
    
    if adv_pool_type == 'seen':
        selected_adv_pool = seen_adv_pool
    elif adv_pool_type == 'unseen':
        selected_adv_pool = unseen_adv_pool
    else:
        selected_adv_pool = mix_adv_pool
    
    adv_path = [None] * len(selected_adv_pool)

    all_policies_vae = get_three_adv_all_policies(env_vae, selected_adv_pool, adv_path)
    all_policies_bandit = get_three_adv_all_policies(env_bandit, selected_adv_pool, adv_path)
    
    opponent_model = OpponentModel(16, 4, hidden_dim, embedding_dim, 7, encoder_weight_path, decoder_weight_path)
    vi = VariationalInference(opponent_model, latent_dim=embedding_dim, n_update_times=10, game_steps=args.steps)

    # bandit use EXP3
    opponent_model_bandit = OpponentModel(16, 4, hidden_dim, embedding_dim, 7, encoder_weight_path, decoder_weight_path)
    vi_bandit = VariationalInference(opponent_model_bandit, latent_dim=embedding_dim, n_update_times=10, game_steps=args.steps)
    exp3 = EXP3(n_action=2, gamma=0.2, min_reward=-200, max_reward=20)

    use_exp3 = True
    
    cur_adv_idx = 10 * exp_num + 200
    # cur_adv_idx = 0

    for i_episode in range(num_episodes):
        if i_episode % adv_change_freq == 0:
            policies_vae = []
            policies_bandit = []

            policy_vec = policy_vec_seq[cur_adv_idx]
            print(policy_vec)
            for j in range(N_ADV):
                adv_idx = np.argmax(policy_vec)
                policies_vae.append(all_policies_vae[j][adv_idx])
                policies_bandit.append(all_policies_bandit[j][adv_idx])
            cur_adv_idx += 1
        
        episode_return_vae = 0
        episode_return_bandit = 0

        obs_n_vae = env_vae._reset()
        obs_n_bandit = env_bandit._reset()

        obs_adv = []
        act_adv = []
        obs_adv_bandit = []
        act_adv_bandit = []
        if use_exp3:
            agent_selected = exp3.sample_action()
        else:
            agent_selected = 1

        for st in range(args.steps):
            act_n_vae = []
            act_n_bandit = []
        
            # vae
            for j, policy in enumerate(policies_vae):
                act_vae = policy.action(obs_n_vae[j])
                act_n_vae.append(act_vae)
            cur_latent = vi.generate_cur_embedding(is_np=False).to(device)
            obs_adv.append(obs_n_vae[0])
            act_adv.append(act_n_vae[0])

            act_vae,_,_ = agent_vae.select_action(obs_n_vae[3], cur_latent, 2)
            act_n_vae.append(act_vae)
            next_obs_n_vae, reward_n_vae,_,_ = env_vae._step(act_n_vae)
            episode_return_vae += reward_n_vae[3]
            obs_n_vae = next_obs_n_vae

            # bandit
            for j, policy in enumerate(policies_bandit):
                act_bandit = policy.action(obs_n_bandit[j])
                act_n_bandit.append(act_bandit)
            cur_latent_bandit = vi_bandit.generate_cur_embedding(is_np=False).to(device)
            obs_adv_bandit.append(obs_n_bandit[0])
            act_adv_bandit.append(act_n_bandit[0])

            if agent_selected:
                act_bandit = agent_bandit_pi.action(obs_n_bandit[3])
            else:
                act_bandit,_,_ = agent_vae.select_action(obs_n_bandit[3], cur_latent_bandit, 2)
            act_n_bandit.append(act_bandit)
            next_obs_n_bandit, reward_n_bandit, done_n_bandit, _ = env_bandit._step(act_n_bandit)
            episode_return_bandit += reward_n_bandit[3]
            obs_n_bandit = next_obs_n_bandit


        if i_episode % ckp_freq == 0:
            print('current episode', i_episode)
            play_episodes = 100
            return_list_vae = return_list_vae + play_multiple_times_without_update_simple(env_vae, agent_vae, policies_vae[0], policies_vae[1], policies_vae[2], 'vae', 'rule', play_episodes=play_episodes, latent=cur_latent)
            print('avg reward of vae:', sum(return_list_vae[-100:])/100)
            return_list_bandit = return_list_bandit + play_multiple_times_without_update_bandit(env_bandit, agent_bandit_pi, agent_vae, policies_bandit[0], policies_bandit[1], policies_bandit[2], 'rule', play_episodes=play_episodes, latent=cur_latent_bandit, p=exp3.get_p()[0], use_exp3=use_exp3)
            print('avg reward of bandit', sum(return_list_bandit[-100:])/100)
            
            vi_emb = vi.generate_cur_embedding(is_np=True)
            vi_ces = vi.get_cur_ce()
            print("vae embedding", vi_emb)
            print("vae cross entropy loss", vi_ces)

            result_dict = {}
            result_dict['num_episodes'] = len(return_list_vae)
            result_dict['return_list_vae'] = return_list_vae
            result_dict['return_list_bandit'] = return_list_bandit
            
            pickle.dump(result_dict, open('results/online_adaption_' + test_id + '.p', "wb"))
            

        act_adv = np.array(act_adv)
        act_adv = np.argmax(act_adv.astype(np.float32), axis=1)
        obs_adv_tensor = torch.FloatTensor(obs_adv)
        act_adv_tensor = torch.FloatTensor(act_adv)
        vi.update(obs_adv_tensor, act_adv_tensor)

    
        act_adv_bandit = np.array(act_adv_bandit)
        act_adv_bandit = np.argmax(act_adv_bandit.astype(np.float32), axis=1)
        obs_adv_tensor_bandit = torch.FloatTensor(obs_adv_bandit)
        act_adv_tensor_bandit = torch.FloatTensor(act_adv_bandit)
        vi_bandit.update(obs_adv_tensor_bandit, act_adv_tensor_bandit)

        if use_exp3:
            exp3.update(episode_return_bandit, agent_selected)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l1', '--lr1', default=1e-4, help='Actor learning rate')
    parser.add_argument('-l2', '--lr2', default=1e-4, help='Critic learning rate')
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, help='Num of steps in a single run')
    parser.add_argument('-ep', '--num_episodes', default=1000, help='Num of episodes')
    args = parser.parse_args()

    main(args)
