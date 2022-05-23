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
from VAE.opponent_models import OpponentModel
from VAE.bayesian_update import BayesianUpdater, VariationalInference, EXP3
from conditional_RL.GSCU_Greedy import PPO_VAE
from conditional_RL.simple_ppo import PPO
from online_test.multiple_test import *

N_ADV = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def main(args):
    gamma = 0.99
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
    hidden_dim = 128
    seed = 1
    actor_lr = args.lr1
    critic_lr = args.lr2

    num_episodes = args.num_episodes
    
    # path option: RL1_10000.pt, RL2_10000.pt, or other ppo weight path as you wish
    adv_init_path = '../conditional_RL/trained_parameters/PPO_adv/RL2_10000.pt'
    exp_id = 'v57' # where model param init from
    ckp_num = 10000
    ckp_freq = 50
    test_id = 'v10'
    print(test_id)

    settings = {}
    settings['adv_init_path'] = adv_init_path
    settings['params_exp_id'] = exp_id
    settings['seed'] = seed
    
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
    print(settings)

    # PPO_advs:
    state_dim_adv = env_VAE.observation_space[0].shape[0]
    action_dim_adv = env_VAE.action_space[0].n
    adv_vae = PPO(state_dim_adv, hidden_dim, action_dim_adv, actor_lr, critic_lr, gamma)
    adv_pi = PPO(state_dim_adv, hidden_dim, action_dim_adv, actor_lr, critic_lr, gamma)
    adv_bandit = PPO(state_dim_adv, hidden_dim, action_dim_adv, actor_lr, critic_lr, gamma)
    
    adv_vae.init_from_save(adv_init_path)
    adv_pi.init_from_save(adv_init_path)
    adv_bandit.init_from_save(adv_init_path)
    

    # add for vae embedding generation
    opponent_model = OpponentModel(16, 4, hidden_dim, embedding_dim, 7, encoder_weight_path, decoder_weight_path)
    vi = VariationalInference(opponent_model, latent_dim=embedding_dim, n_update_times=10, game_steps=args.steps)

    # bandit use EXP3
    opponent_model_bandit = OpponentModel(16, 4, hidden_dim, embedding_dim, 7, encoder_weight_path, decoder_weight_path)
    vi_bandit = VariationalInference(opponent_model_bandit, latent_dim=embedding_dim, n_update_times=10, game_steps=args.steps)
    exp3 = EXP3(n_action=2, gamma=0.2, min_reward=-200, max_reward=100)

    use_exp3 = True
    
    for i_episode in range(num_episodes):
        episode_return_vae = 0
        episode_return_pi = 0
        episode_return_bandit = 0

        obs_n_vae = env_VAE._reset()
        obs_n_pi = env_pi._reset()
        obs_n_bandit = env_bandit._reset()

        # add for vae
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

            # advs
            act_adv_1_pi, act_index_adv_1_pi, act_prob_adv_1_pi = adv_pi.select_action(obs_n_pi[0], 2)
            act_adv_2_pi, act_index_adv_2_pi, act_prob_adv_2_pi = adv_pi.select_action(obs_n_pi[1], 2)
            act_adv_3_pi, act_index_adv_3_pi, act_prob_adv_3_pi = adv_pi.select_action(obs_n_pi[2], 2)
            act_n_pi.append(act_adv_1_pi)
            act_n_pi.append(act_adv_2_pi)
            act_n_pi.append(act_adv_3_pi)

            act_adv_1_vae_, act_index_adv_1_vae, act_prob_adv_1_vae = adv_vae.select_action(obs_n_vae[0], 2)
            act_adv_2_vae_, act_index_adv_2_vae, act_prob_adv_2_vae = adv_vae.select_action(obs_n_vae[1], 2)
            act_adv_3_vae_, act_index_adv_3_vae, act_prob_adv_3_vae = adv_vae.select_action(obs_n_vae[2], 2)
            act_n_vae.append(act_adv_1_vae_)
            act_n_vae.append(act_adv_2_vae_)
            act_n_vae.append(act_adv_3_vae_)

            act_adv_1_bandit_, act_index_adv_1_bandit, act_prob_adv_1_bandit = adv_bandit.select_action(obs_n_bandit[0], 2)
            act_adv_2_bandit_, act_index_adv_2_bandit, act_prob_adv_2_bandit = adv_bandit.select_action(obs_n_bandit[1], 2)
            act_adv_3_bandit_, act_index_adv_3_bandit, act_prob_adv_3_bandit = adv_bandit.select_action(obs_n_bandit[2], 2)
            act_n_bandit.append(act_adv_1_bandit_)
            act_n_bandit.append(act_adv_2_bandit_)
            act_n_bandit.append(act_adv_3_bandit_)

            # pi_1^*
            act_pi = agent_pi.action(obs_n_pi[3])
            act_n_pi.append(act_pi)
            next_obs_n_pi, reward_n_pi, _,_ = env_pi._step(act_n_pi)
            
            trans1_pi = Transition(obs_n_pi[0], act_index_adv_1_pi, act_prob_adv_1_pi, reward_n_pi[0], next_obs_n_pi[0])
            trans2_pi = Transition(obs_n_pi[1], act_index_adv_2_pi, act_prob_adv_2_pi, reward_n_pi[1], next_obs_n_pi[1])
            trans3_pi = Transition(obs_n_pi[2], act_index_adv_3_pi, act_prob_adv_3_pi, reward_n_pi[2], next_obs_n_pi[2])
            adv_pi.store_transition(trans1_pi)
            adv_pi.store_transition(trans2_pi)
            adv_pi.store_transition(trans3_pi)

            episode_return_pi += reward_n_pi[3]
            obs_n_pi = next_obs_n_pi

            # ppo_vae:
            cur_latent = vi.generate_cur_embedding(is_np=False).to(device)
            obs_adv.append(obs_n_vae[0])
            act_adv.append(act_n_vae[0])
    
            act_vae, act_index_vae, act_prob_vae = agent_VAE.select_action(obs_n_vae[env_VAE.n-1], cur_latent, 2)
            act_n_vae.append(act_vae)
            next_obs_n_vae, reward_n_vae, done_n_vae, _ = env_VAE._step(act_n_vae)
            
            trans1_vae = Transition(obs_n_vae[0], act_index_adv_1_vae, act_prob_adv_1_vae, reward_n_vae[0], next_obs_n_vae[0])
            trans2_vae = Transition(obs_n_vae[1], act_index_adv_2_vae, act_prob_adv_2_vae, reward_n_vae[1], next_obs_n_vae[1])
            trans3_vae = Transition(obs_n_vae[2], act_index_adv_3_vae, act_prob_adv_3_vae, reward_n_vae[2], next_obs_n_vae[2])
            adv_vae.store_transition(trans1_vae)
            adv_vae.store_transition(trans2_vae)
            adv_vae.store_transition(trans3_vae)

            episode_return_vae += reward_n_vae[env_VAE.n-1]
            obs_n_vae = next_obs_n_vae

            # bandit
            cur_latent_bandit = vi_bandit.generate_cur_embedding(is_np=False).to(device)
            obs_adv_bandit.append(obs_n_bandit[0])
            act_adv_bandit.append(act_n_bandit[0])
            if agent_selected:
                act_bandit = agent_bandit_pi.action(obs_n_bandit[3])
            else:
                act_bandit,_,_ = agent_VAE.select_action(obs_n_bandit[3], cur_latent_bandit, 2)
            act_n_bandit.append(act_bandit)
            next_obs_n_bandit, reward_n_bandit, done_n_bandit, _ = env_bandit._step(act_n_bandit)
            
            trans1_bandit = Transition(obs_n_bandit[0], act_index_adv_1_bandit, act_prob_adv_1_bandit, reward_n_bandit[0], next_obs_n_bandit[0])
            trans2_bandit = Transition(obs_n_bandit[1], act_index_adv_2_bandit, act_prob_adv_2_bandit, reward_n_bandit[1], next_obs_n_bandit[1])
            trans3_bandit = Transition(obs_n_bandit[2], act_index_adv_3_bandit, act_prob_adv_3_bandit, reward_n_bandit[2], next_obs_n_bandit[2])
            adv_bandit.store_transition(trans1_bandit)
            adv_bandit.store_transition(trans2_bandit)
            adv_bandit.store_transition(trans3_bandit)

            episode_return_bandit += reward_n_bandit[3]
            obs_n_bandit = next_obs_n_bandit


        if i_episode % ckp_freq == 0:
            print('current episode', i_episode)
            play_episodes = 100
            return_list_vae = return_list_vae + play_multiple_times_without_update_simple(env_VAE, agent_VAE, adv_vae, adv_vae, adv_vae, 'vae', 'rl', play_episodes=play_episodes, latent=cur_latent)
            return_list_pi = return_list_pi + play_multiple_times_without_update_simple(env_pi, agent_pi, adv_pi, adv_pi, adv_pi, 'rule', 'rl', play_episodes=play_episodes)
            return_list_bandit = return_list_bandit + play_multiple_times_without_update_bandit(env_bandit, agent_bandit_pi, agent_VAE, adv_bandit, adv_bandit, adv_bandit, 'rl', play_episodes=play_episodes, latent=cur_latent_bandit, p=exp3.get_p()[0], use_exp3=use_exp3)
            
            result_dict = {}
            result_dict['version'] = exp_id
            result_dict['num_episodes'] = len(return_list_vae)
            result_dict['return_list_vae'] = return_list_vae
            result_dict['return_list_pi'] = return_list_pi
            result_dict['return_list_bandit'] = return_list_bandit
            result_dict['settings'] = settings
            
            pickle.dump(result_dict, open('results/PPO_adv_test_'+test_id+'.p', "wb"))
            print('recent average reward of GSCU-Greedy', float(sum(return_list_vae[-100:])/100))
            print('recent average reward of pi_1^*', float(sum(return_list_pi[-100:])/100))
            print('recent average reward of GSCU', float(sum(return_list_bandit[-100:])/100))

        if len(adv_vae.buffer) >= 1500:
            adv_vae.update(i_episode)
            adv_pi.update(i_episode)
            adv_bandit.update(i_episode)
        
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
        
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l1', '--lr1', default=1e-4, help='Actor learning rate')
    parser.add_argument('-l2', '--lr2', default=1e-4, help='Critic learning rate')
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, help='Num of steps in a single run')
    parser.add_argument('-ep', '--num_episodes', default=1500, help='Num of episodes')
    args = parser.parse_args()

    main(args)

