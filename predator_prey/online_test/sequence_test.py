import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
from collections import namedtuple
import pickle
import torch

from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *
import multiagent.scenarios as scenarios
from VAE.opponent_models import *
from VAE.data_generation import get_all_adv_policies
from VAE.opponent_models import OpponentModel
from VAE.bayesian_update import VariationalInference, EXP3, SwitchBoard
from conditional_RL.GSCU_Greedy import PPO_VAE


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
    
    exp_id = 'v57'
    ckp_num = 10000
    exp_num = 1
    test_num = 19
    test_id = 'v' + str(exp_num) + '_' + adv_pool_type + '_' + str(test_num)
    print(test_id)
    
    scenario = scenarios.load(args.scenario).Scenario()
    world_VAE = scenario.make_world()
    world_pi = scenario.make_world()
    world_bandit = scenario.make_world()
    

    env_VAE = MultiAgentEnv(world_VAE, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None, shared_viewer=False, discrete_action=True)
    env_pi = MultiAgentEnv(world_pi, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None, shared_viewer=False, discrete_action=True)
    env_bandit = MultiAgentEnv(world_bandit, scenario.reset_world, scenario.reward, scenario.observation,
                                info_callback=None, shared_viewer=False, discrete_action=True)
    env_VAE.seed(seed)
    env_pi.seed(seed)
    env_bandit.seed(seed)
    np.random.seed(seed)

    state_dim = env_VAE.observation_space[env_VAE.n-1].shape[0]
    action_dim = env_VAE.action_space[env_VAE.n-1].n
    
    embedding_dim = 2
    encoder_weight_path = '../VAE/saved_model_params/encoder_vae_param.pt'
    decoder_weight_path = '../VAE/saved_model_params/decoder_param.pt'

    
    agent_VAE = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, gamma, 4)
    agent_pi = ConservativePolicy(env_pi, env_pi.n-1, epsilon=0.2)
    agent_bandit_pi = ConservativePolicy(env_bandit, env_bandit.n-1, epsilon=0.2)
     
    ppo_vae_path = '../conditional_RL/trained_parameters/GSCU_Greedy/params_' + exp_id + '_' + str(ckp_num) + '.0.pt'
    agent_VAE.init_from_save(ppo_vae_path)
    return_list_vae = []
    return_list_pi = []
    return_list_bandit = []
    
    if adv_pool_type == 'seen':
        selected_adv_pool = seen_adv_pool
    elif adv_pool_type == 'unseen':
        selected_adv_pool = unseen_adv_pool
    else:
        selected_adv_pool = mix_adv_pool
    adv_path = [None] * len(selected_adv_pool)
    
    
    all_policies_vae = get_three_adv_all_policies(env_VAE, selected_adv_pool, adv_path)
    all_policies_pi = get_three_adv_all_policies(env_pi, selected_adv_pool, adv_path)
    all_policies_bandit = get_three_adv_all_policies(env_bandit, selected_adv_pool, adv_path)

    # add for vae embedding generation
    opponent_model = OpponentModel(16, 4, hidden_dim, embedding_dim, 7, encoder_weight_path, decoder_weight_path)
    vi = VariationalInference(opponent_model, latent_dim=embedding_dim, n_update_times=10, game_steps=args.steps)

    # bandit use EXP3
    opponent_model_bandit = OpponentModel(16, 4, hidden_dim, embedding_dim, 7, encoder_weight_path, decoder_weight_path)
    vi_bandit = VariationalInference(opponent_model_bandit, latent_dim=embedding_dim, n_update_times=10, game_steps=args.steps)
    exp3 = EXP3(n_action=2, gamma=0.2, min_reward=-200, max_reward=50)

    use_exp3 = True

    cur_adv_idx = 20 * exp_num + 300
    for i_episode in range(num_episodes):
        # change opponent policy
        if (i_episode % adv_change_freq == 0):
            policies_vae = []
            policies_pi = []
            policies_bandit = []
            policy_vec = policy_vec_seq[cur_adv_idx]
            
            print(policy_vec)
            for j in range(N_ADV):
                adv_idx = np.argmax(policy_vec)
                policies_vae.append(all_policies_vae[j][adv_idx])
                policies_pi.append(all_policies_pi[j][adv_idx])
                policies_bandit.append(all_policies_bandit[j][adv_idx])
            cur_adv_idx += 1
        
        
        episode_return_vae = 0
        episode_return_pi = 0
        episode_return_bandit = 0

        obs_n_vae = env_VAE._reset()
        obs_n_pi = env_pi._reset()
        obs_n_bandit = env_bandit._reset()

        # add for vae and bandit
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
            act_n_pi = []
            act_n_bandit = []

            # pi_1^*
            for j, policy in enumerate(policies_pi):
                act_pi = policy.action(obs_n_pi[j])
                act_n_pi.append(act_pi)
            act_pi = agent_pi.action(obs_n_pi[3])
            act_n_pi.append(act_pi)
            next_obs_n_pi, reward_n_pi, _,_ = env_pi._step(act_n_pi)
            episode_return_pi += reward_n_pi[3]
            obs_n_pi = next_obs_n_pi

            # ppo_vae:
            for j, policy in enumerate(policies_vae):
                act_vae = policy.action(obs_n_vae[j])
                act_n_vae.append(act_vae)

            cur_latent = vi.generate_cur_embedding(is_np=False).to(device)
            obs_adv.append(obs_n_vae[0])
            act_adv.append(act_n_vae[0])
    
            act_vae, act_index_vae, act_prob_vae = agent_VAE.select_action(obs_n_vae[env_VAE.n-1], cur_latent, env_VAE.world.dim_c)
            act_n_vae.append(act_vae)
            next_obs_n_vae, reward_n_vae, done_n_vae, _ = env_VAE._step(act_n_vae)
            episode_return_vae += reward_n_vae[env_VAE.n-1]
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
                act_bandit,_,_ = agent_VAE.select_action(obs_n_bandit[3], cur_latent_bandit, 2)
            act_n_bandit.append(act_bandit)
            next_obs_n_bandit, reward_n_bandit, done_n_bandit, _ = env_bandit._step(act_n_bandit)
            episode_return_bandit += reward_n_bandit[3]
            obs_n_bandit = next_obs_n_bandit          

    
        return_list_vae.append(episode_return_vae)
        return_list_pi.append(episode_return_pi)
        return_list_bandit.append(episode_return_bandit)
        
        if (i_episode % 50 == 0 and i_episode > 0):
            print('current episode', i_episode)
            print('recent avg return of GSCU-Greedy', sum(return_list_vae[-50:])/50)
            print('recent avg return of $\pi_1^*$', sum(return_list_pi[-50:])/50)
            print('recent avg return of GSCU', sum(return_list_bandit[-50:])/50)
            
        
        # update vae embedding generation
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
        

    result_dict = {}
    result_dict['version'] = exp_id
    result_dict['num_episodes'] = len(return_list_vae)
    result_dict['return_list_vae'] = return_list_vae
    result_dict['return_list_pi'] = return_list_pi
    result_dict['return_list_bandit'] = return_list_bandit
    
    pickle.dump(result_dict, open('results/sequence_test_'+test_id+'.p', "wb"))

    print('average reward of ppo_vae', float(sum(return_list_vae)/4000))
    print('average reward of pi_1^*', float(sum(return_list_pi)/4000))
    print('average reward of bandit', float(sum(return_list_bandit)/4000))
    

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l1', '--lr1', default=1e-4, help='Actor learning rate')
    parser.add_argument('-l2', '--lr2', default=1e-4, help='Critic learning rate')
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, help='Num of steps in a single run')
    parser.add_argument('-ep', '--num_episodes', default=4000, help='Num of episodes')
    args = parser.parse_args()

    main(args)

