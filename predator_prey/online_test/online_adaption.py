import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import pickle
import torch
import logging

from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *
import multiagent.scenarios as scenarios
from embedding_learning.opponent_models import OpponentModel
from embedding_learning.data_generation import get_all_adv_policies
from online_test.bayesian_update import VariationalInference, EXP3
from conditional_RL.conditional_rl_model import PPO_VAE
from conditional_RL.ppo_model import PPO
from utils.multiple_test import *
from utils.config_predator_prey import Config

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

N_ADV = 3
seen_adv_pool = Config.ADV_POOL_SEEN
unseen_adv_pool =  Config.ADV_POOL_UNSEEN
mix_adv_pool = Config.ADV_POOL_MIX
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_three_adv_all_policies(env, adv_pool):
    all_policies_idx0 = get_all_adv_policies(env,adv_pool,agent_index=0)
    all_policies_idx1 = get_all_adv_policies(env,adv_pool,agent_index=1)
    all_policies_idx2 = get_all_adv_policies(env,adv_pool,agent_index=2)
    all_policies = [all_policies_idx0,all_policies_idx1,all_policies_idx2]
    return all_policies

def main(args):
    gamma = 0.99
    hidden_dim = Config.HIDDEN_DIM
    seed = args.seed
    adv_change_freq = 200
    n_opponent_per_seq = 20
    n_episode_seq = adv_change_freq * n_opponent_per_seq
    n_seq = 5
    num_episodes = n_episode_seq * n_seq

    window_size = Config.WINDOW_SIZW
    adv_pool_type = args.opp_type

    rst_dir = Config.ONLINE_TEST_RST_DIR
    data_dir = Config.DATA_DIR
    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir, exist_ok=False) 

    if adv_pool_type == 'mix':
        dataloader = open(data_dir+'online_test_policy_vec_seq_8.p', 'rb')
    elif adv_pool_type == 'seen' or adv_pool_type == 'unseen':
        dataloader = open(data_dir+'online_test_policy_vec_seq_4.p', 'rb')
    else:
        print('Please choose seen/unseen/mix')
    data = pickle.load(dataloader)
    policy_vec_seq = data['policy_vec_seq']

    ckp_freq = 20
    test_id = args.version

    scenario = scenarios.load(args.scenario).Scenario()
    world_pi = scenario.make_world()
    world_vae = scenario.make_world()
    world_bandit = scenario.make_world()

    env_pi = MultiAgentEnv(world_pi, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None, shared_viewer=False, discrete_action=True)
    env_vae = MultiAgentEnv(world_vae, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None, shared_viewer=False, discrete_action=True)
    env_bandit = MultiAgentEnv(world_bandit, scenario.reset_world, scenario.reward, scenario.observation,
                                info_callback=None, shared_viewer=False, discrete_action=True)
    env_pi.seed(seed)
    env_vae.seed(seed)
    env_bandit.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    state_dim = env_vae.observation_space[3].shape[0]
    action_dim = env_vae.action_space[3].n
    embedding_dim = Config.LATENT_DIM
    encoder_weight_path = args.encoder_file
    decoder_weight_path = args.decoder_file

    agent_vae = PPO_VAE(state_dim+action_dim, hidden_dim, embedding_dim, action_dim, 0.0, 0.0, encoder_weight_path, gamma, 4)
    agent_pi = PPO(state_dim, hidden_dim, action_dim, 0.0, 0.0, gamma)
    ppo_vae_path = args.rl_file
    ppo_pi_path = '../model_params/RL/params_pi.pt'
    agent_vae.init_from_save(ppo_vae_path)
    agent_pi.init_from_save(ppo_pi_path)
    return_list_vae = []
    return_list_bandit = []
    
    if adv_pool_type == 'seen':
        selected_adv_pool = seen_adv_pool
    elif adv_pool_type == 'unseen':
        selected_adv_pool = unseen_adv_pool
    else:
        selected_adv_pool = mix_adv_pool
    
    all_policies_pi = get_three_adv_all_policies(env_pi, selected_adv_pool)
    all_policies_vae = get_three_adv_all_policies(env_vae, selected_adv_pool)
    all_policies_bandit = get_three_adv_all_policies(env_bandit, selected_adv_pool)
    
    opponent_model = OpponentModel(16, 4, hidden_dim, embedding_dim, action_dim+2, encoder_weight_path, decoder_weight_path)
    vi = VariationalInference(opponent_model, latent_dim=embedding_dim, n_update_times=10, game_steps=args.steps)

    # GSCU use EXP3
    opponent_model_bandit = OpponentModel(16, 4, hidden_dim, embedding_dim, action_dim+2, encoder_weight_path, decoder_weight_path)
    vi_bandit = VariationalInference(opponent_model_bandit, latent_dim=embedding_dim, game_steps=args.steps)
    exp3 = EXP3(n_action=2, gamma=0.2, min_reward=-200, max_reward=20)

    use_exp3 = True
    cur_adv_idx = 300 # just a random number to indetify the sequence start point. can be any number between 0 to 800
    cur_n_opponent = 0

    return_list_vae = []
    return_list_pi = []
    return_list_bandit = []

    for i_episode in range(num_episodes):

        if i_episode % n_episode_seq == 0:
            exp3.init_weight()
            vi.init_all() 
            vi_bandit.init_all() 

        if i_episode % adv_change_freq == 0:
            policies_pi = []
            policies_vae = []
            policies_bandit = []

            policy_vec = policy_vec_seq[cur_adv_idx]
            for j in range(N_ADV):
                adv_idx = np.argmax(policy_vec)
                policies_pi.append(all_policies_pi[j][adv_idx])
                policies_vae.append(all_policies_vae[j][adv_idx])
                policies_bandit.append(all_policies_bandit[j][adv_idx])
            opp_name = selected_adv_pool[adv_idx]
        
        episode_return_pi = 0
        episode_return_vae = 0
        episode_return_bandit = 0

        obs_n_pi = env_pi._reset()
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

        obs_traj_vae = [np.zeros(state_dim)]*(window_size-1)
        act_traj_vae = [np.zeros(action_dim)]*window_size
        hidden_vae = [torch.zeros((1,1,hidden_dim)).to(device), torch.zeros((1,1,hidden_dim)).to(device)]

        obs_traj_bandit = [np.zeros(state_dim)]*(window_size-1)
        act_traj_bandit = [np.zeros(action_dim)]*window_size
        hidden_bandit = [torch.zeros((1,1,hidden_dim)).to(device), torch.zeros((1,1,hidden_dim)).to(device)]

        for st in range(args.steps):
            act_n_pi = []
            act_n_vae = []
            act_n_bandit = []

            # pi_1^*
            for j, policy in enumerate(policies_pi):
                act_pi = policy.action(obs_n_pi[j])
                act_n_pi.append(act_pi)
            act_pi,_,_ = agent_pi.select_action(obs_n_pi[3], 2)
            # act_pi = agent_pi.action(obs_n_pi[3])
            act_n_pi.append(act_pi)
            next_obs_n_pi, reward_n_pi, _,_ = env_pi._step(act_n_pi)
            episode_return_pi += reward_n_pi[env_pi.n-1]
            obs_n_pi = next_obs_n_pi

            # GSCU-Greedy
            for j, policy in enumerate(policies_vae):
                act_vae = policy.action(obs_n_vae[j])
                act_n_vae.append(act_vae)
            cur_latent = vi.generate_cur_embedding(is_np=False).to(device)
            obs_adv.append(obs_n_vae[0])
            act_adv.append(act_n_vae[0])
            obs_traj_vae.append(obs_n_vae[3])
            obs_traj_tensor_vae = torch.tensor([obs_traj_vae], dtype=torch.float).to(device)
            act_traj_tensor_vae = torch.tensor([act_traj_vae], dtype=torch.float).to(device)
            act_vae, act_index_vae, act_prob_vae = agent_vae.select_action(obs_traj_tensor_vae, act_traj_tensor_vae, hidden_vae, cur_latent, env_vae.world.dim_c)
            act_n_vae.append(act_vae)
            next_obs_n_vae, reward_n_vae, done_n_vae, _ = env_vae._step(act_n_vae)
            episode_return_vae += reward_n_vae[env_vae.n-1]
            obs_n_vae = next_obs_n_vae
            if len(obs_traj_vae) >= window_size:
                obs_traj_vae.pop(0)
                act_traj_vae.pop(0)
            act_traj_vae.append(act_vae[:-2])

            # GSCU (w/ bandit)
            for j, policy in enumerate(policies_bandit):
                act_bandit = policy.action(obs_n_bandit[j])
                act_n_bandit.append(act_bandit)
            cur_latent_bandit = vi_bandit.generate_cur_embedding(is_np=False).to(device)
            obs_adv_bandit.append(obs_n_bandit[0])
            act_adv_bandit.append(act_n_bandit[0])
            obs_traj_bandit.append(obs_n_bandit[3])
            obs_traj_tensor_bandit = torch.tensor([obs_traj_bandit], dtype=torch.float).to(device)
            act_traj_tensor_bandit = torch.tensor([act_traj_bandit], dtype=torch.float).to(device)
            if agent_selected:
                act_bandit,_,_ = agent_pi.select_action(obs_n_bandit[3], 2)
            else:
                act_bandit,_,_ = agent_vae.select_action(obs_traj_tensor_bandit, act_traj_tensor_bandit, hidden_bandit, cur_latent_bandit, 2)
            act_n_bandit.append(act_bandit)
            next_obs_n_bandit, reward_n_bandit, done_n_bandit, _ = env_bandit._step(act_n_bandit)
            episode_return_bandit += reward_n_bandit[3]
            obs_n_bandit = next_obs_n_bandit
            if len(obs_traj_bandit) >= window_size:
                obs_traj_bandit.pop(0)
                act_traj_bandit.pop(0)
            act_traj_bandit.append(act_bandit[:-2])


        return_list_vae.append(episode_return_vae)
        return_list_pi.append(episode_return_pi)
        return_list_bandit.append(episode_return_bandit)


        seq_idx = cur_n_opponent//n_opponent_per_seq
        opp_idx = cur_n_opponent%n_opponent_per_seq

        if (i_episode+1)%adv_change_freq == 0:
            logging.info("seq idx: {}, opp idx: {}, opp name: {}, gscu: {:.2f}, | greedy: {:.2f}, | pi: {:.2f}".format(
                        seq_idx,opp_idx,opp_name,np.mean(return_list_bandit[-adv_change_freq:]),np.mean(return_list_vae[-adv_change_freq:]),np.mean(return_list_pi[-adv_change_freq:])))

            if (cur_n_opponent+1)%n_opponent_per_seq == 0:
                print ('# seq: ', seq_idx, ', total # of opp: ', cur_n_opponent+1,
                        ', avg gscu', np.mean(return_list_bandit), 
                        '| avg greedy', np.mean(return_list_vae), 
                        '| avg pi', np.mean(return_list_pi))
                print ('-'*10)

                result_dict = {}
                result_dict['opponent_type'] = adv_pool_type
                result_dict['version'] = test_id
                result_dict['n_opponent'] = cur_n_opponent+1
                result_dict['greedy'] = return_list_vae
                result_dict['pi'] = return_list_pi
                result_dict['gscu'] = return_list_bandit
                pickle.dump(result_dict, open(rst_dir+'online_adaption_'+test_id+'_'+adv_pool_type+'.p', "wb"))


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
        
        if i_episode % adv_change_freq == 0 and i_episode>0:
            cur_adv_idx += 1
            cur_n_opponent += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, help='Num of steps in a single run')
    parser.add_argument('-v', '--version', default='v0', help='version')
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-o', '--opp_type', default='seen', 
                        choices=["seen", "unseen", "mix"], help='type of the opponents')
    parser.add_argument('-e', '--encoder_file', default='../model_params/VAE/encoder_vae_param_demo.pt', help='vae encoder file')
    parser.add_argument('-d', '--decoder_file', default='../model_params/VAE/decoder_param_demo.pt', help='vae decoder file')
    parser.add_argument('-r', '--rl_file', default='../model_params/RL/params_demo.pt', help='conditional RL file')
    args = parser.parse_args()
    

    main(args)
