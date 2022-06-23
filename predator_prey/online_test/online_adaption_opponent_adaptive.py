import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
from collections import namedtuple
import pickle
import torch
import logging

from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *
import multiagent.scenarios as scenarios
from embedding_learning.opponent_models import *
from embedding_learning.opponent_models import OpponentModel
from online_test.bayesian_update import VariationalInference, EXP3
from conditional_RL.conditional_rl_model import PPO_VAE
from conditional_RL.ppo_model import PPO
from utils.multiple_test import *
from utils.config_predator_prey import Config

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

N_ADV = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    gamma = 0.99
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
    hidden_dim = Config.HIDDEN_DIM
    seed = int(args.seed)
    actor_lr = 5e-5
    critic_lr = 5e-5
    num_episodes = 200*5
    ckp_freq = 50
    test_id = args.version
    batch_size = 400 
    n_test = 100
    window_size = Config.WINDOW_SIZW

    rst_dir = Config.ONLINE_TEST_RST_DIR
    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir, exist_ok=False) 

    settings = {}
    settings['opp_init_id'] = args.opp_init_id
    settings['params_exp_id'] = test_id
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    state_dim = env_VAE.observation_space[env_VAE.n-1].shape[0]
    action_dim = env_VAE.action_space[env_VAE.n-1].n
    
    embedding_dim = Config.LATENT_DIM
    encoder_weight_path = args.encoder_file
    decoder_weight_path = args.decoder_file

    encoder_weight_path = Config.VAE_MODEL_DIR + args.encoder_file
    decoder_weight_path = Config.VAE_MODEL_DIR + args.decoder_file
    ppo_vae_path = Config.RL_MODEL_DIR + args.rl_file
    ppo_pi_path = '../model_params/RL/params_pi.pt'
    opponent_weight_path = Config.OPPONENT_MODEL_DIR + 'params_opp_init_' + args.opp_init_id + '.pt'

    agent_VAE = PPO_VAE(state_dim+action_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, gamma, 4)
    agent_pi = PPO(state_dim, hidden_dim, action_dim, 0.0, 0.0, gamma)
    agent_VAE.init_from_save(ppo_vae_path)
    agent_pi.init_from_save(ppo_pi_path)

    return_vae_list = []
    return_pi_list = []
    return_bandit_list = []
    print(settings)

    # PPO_advs:
    state_dim_adv = env_VAE.observation_space[0].shape[0]
    action_dim_adv = env_VAE.action_space[0].n
    opponent_vae = PPO(state_dim_adv, hidden_dim, action_dim_adv, actor_lr, critic_lr, gamma)
    opponent_pi = PPO(state_dim_adv, hidden_dim, action_dim_adv, actor_lr, critic_lr, gamma)
    opponent_bandit = PPO(state_dim_adv, hidden_dim, action_dim_adv, actor_lr, critic_lr, gamma)
    
    opponent_vae.init_from_save(opponent_weight_path)
    opponent_pi.init_from_save(opponent_weight_path)
    opponent_bandit.init_from_save(opponent_weight_path)

    opponent_vae.batch_size = batch_size
    opponent_pi.batch_size = batch_size
    opponent_bandit.batch_size = batch_size
    

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
        obs_opponent_bandit = []
        act_opponent_bandit = []
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
            act_n_vae = []
            act_n_pi = []
            act_n_bandit = []

            # advs
            act_adv_1_pi, act_index_adv_1_pi, act_prob_adv_1_pi = opponent_pi.select_action(obs_n_pi[0], 2)
            act_adv_2_pi, act_index_adv_2_pi, act_prob_adv_2_pi = opponent_pi.select_action(obs_n_pi[1], 2)
            act_adv_3_pi, act_index_adv_3_pi, act_prob_adv_3_pi = opponent_pi.select_action(obs_n_pi[2], 2)
            act_n_pi.append(act_adv_1_pi)
            act_n_pi.append(act_adv_2_pi)
            act_n_pi.append(act_adv_3_pi)

            act_adv_1_vae, act_index_adv_1_vae, act_prob_adv_1_vae = opponent_vae.select_action(obs_n_vae[0], 2)
            act_adv_2_vae, act_index_adv_2_vae, act_prob_adv_2_vae = opponent_vae.select_action(obs_n_vae[1], 2)
            act_adv_3_vae, act_index_adv_3_vae, act_prob_adv_3_vae = opponent_vae.select_action(obs_n_vae[2], 2)
            act_n_vae.append(act_adv_1_vae)
            act_n_vae.append(act_adv_2_vae)
            act_n_vae.append(act_adv_3_vae)

            act_adv_1_bandit, act_index_adv_1_bandit, act_prob_adv_1_bandit = opponent_bandit.select_action(obs_n_bandit[0], 2)
            act_adv_2_bandit, act_index_adv_2_bandit, act_prob_adv_2_bandit = opponent_bandit.select_action(obs_n_bandit[1], 2)
            act_adv_3_bandit, act_index_adv_3_bandit, act_prob_adv_3_bandit = opponent_bandit.select_action(obs_n_bandit[2], 2)
            act_n_bandit.append(act_adv_1_bandit)
            act_n_bandit.append(act_adv_2_bandit)
            act_n_bandit.append(act_adv_3_bandit)

            # pi_1^*
            act_pi,_,_ = agent_pi.select_action(obs_n_pi[3], 2)
            act_n_pi.append(act_pi)
            next_obs_n_pi, reward_n_pi, _,_ = env_pi._step(act_n_pi)
            
            trans1_pi = Transition(obs_n_pi[0], act_index_adv_1_pi, act_prob_adv_1_pi, reward_n_pi[0], next_obs_n_pi[0])
            trans2_pi = Transition(obs_n_pi[1], act_index_adv_2_pi, act_prob_adv_2_pi, reward_n_pi[1], next_obs_n_pi[1])
            trans3_pi = Transition(obs_n_pi[2], act_index_adv_3_pi, act_prob_adv_3_pi, reward_n_pi[2], next_obs_n_pi[2])
            opponent_pi.store_transition(trans1_pi)
            opponent_pi.store_transition(trans2_pi)
            opponent_pi.store_transition(trans3_pi)

            episode_return_pi += reward_n_pi[3]
            obs_n_pi = next_obs_n_pi

            # ppo_vae:
            cur_latent = vi.generate_cur_embedding(is_np=False).to(device)
            obs_adv.append(obs_n_vae[0])
            act_adv.append(act_n_vae[0])
            obs_traj_vae.append(obs_n_vae[3])
            obs_traj_tensor_vae = torch.tensor([obs_traj_vae], dtype=torch.float).to(device)
            act_traj_tensor_vae = torch.tensor([act_traj_vae], dtype=torch.float).to(device)
            act_vae, act_index_vae, act_prob_vae = agent_VAE.select_action(obs_traj_tensor_vae, act_traj_tensor_vae, hidden_vae, cur_latent, env_VAE.world.dim_c)
            act_n_vae.append(act_vae)
            next_obs_n_vae, reward_n_vae, done_n_vae, _ = env_VAE._step(act_n_vae)
            
            trans1_vae = Transition(obs_n_vae[0], act_index_adv_1_vae, act_prob_adv_1_vae, reward_n_vae[0], next_obs_n_vae[0])
            trans2_vae = Transition(obs_n_vae[1], act_index_adv_2_vae, act_prob_adv_2_vae, reward_n_vae[1], next_obs_n_vae[1])
            trans3_vae = Transition(obs_n_vae[2], act_index_adv_3_vae, act_prob_adv_3_vae, reward_n_vae[2], next_obs_n_vae[2])
            opponent_vae.store_transition(trans1_vae)
            opponent_vae.store_transition(trans2_vae)
            opponent_vae.store_transition(trans3_vae)

            episode_return_vae += reward_n_vae[env_VAE.n-1]
            obs_n_vae = next_obs_n_vae
            if len(obs_traj_vae) >= window_size:
                obs_traj_vae.pop(0)
                act_traj_vae.pop(0)
            act_traj_vae.append(act_vae[:-2])

            # bandit
            cur_latent_bandit = vi_bandit.generate_cur_embedding(is_np=False).to(device)
            obs_opponent_bandit.append(obs_n_bandit[0])
            act_opponent_bandit.append(act_n_bandit[0])
            obs_traj_bandit.append(obs_n_bandit[3])
            obs_traj_tensor_bandit = torch.tensor([obs_traj_bandit], dtype=torch.float).to(device)
            act_traj_tensor_bandit = torch.tensor([act_traj_bandit], dtype=torch.float).to(device)
            if agent_selected:
                act_bandit,_,_ = agent_pi.select_action(obs_n_bandit[3], 2)
            else:
                act_bandit,_,_ = agent_VAE.select_action(obs_traj_tensor_bandit, act_traj_tensor_bandit, hidden_bandit, cur_latent_bandit, 2)
            act_n_bandit.append(act_bandit)
            next_obs_n_bandit, reward_n_bandit, done_n_bandit, _ = env_bandit._step(act_n_bandit)
            
            trans1_bandit = Transition(obs_n_bandit[0], act_index_adv_1_bandit, act_prob_adv_1_bandit, reward_n_bandit[0], next_obs_n_bandit[0])
            trans2_bandit = Transition(obs_n_bandit[1], act_index_adv_2_bandit, act_prob_adv_2_bandit, reward_n_bandit[1], next_obs_n_bandit[1])
            trans3_bandit = Transition(obs_n_bandit[2], act_index_adv_3_bandit, act_prob_adv_3_bandit, reward_n_bandit[2], next_obs_n_bandit[2])
            opponent_bandit.store_transition(trans1_bandit)
            opponent_bandit.store_transition(trans2_bandit)
            opponent_bandit.store_transition(trans3_bandit)

            episode_return_bandit += reward_n_bandit[3]
            obs_n_bandit = next_obs_n_bandit
            if len(obs_traj_bandit) >= window_size:
                obs_traj_bandit.pop(0)
                act_traj_bandit.pop(0)
            act_traj_bandit.append(act_bandit[:-2])


        if i_episode % ckp_freq == 0:
            ckp_return_vae_list = play_multiple_times_test(env_VAE, agent_VAE, opponent_vae, opponent_vae, opponent_vae, 'vae', 'rl', play_episodes=n_test, latent=cur_latent)
            ckp_return_pi_list = play_multiple_times_test(env_pi, agent_pi, opponent_pi, opponent_pi, opponent_pi, 'ppo', 'rl', play_episodes=n_test)
            ckp_eturn_list_bandit = play_multiple_times_test_bandit(env_bandit, agent_pi, agent_VAE, opponent_bandit, opponent_bandit, opponent_bandit, 'rl', play_episodes=n_test, latent=cur_latent_bandit, p=exp3.get_p()[0], use_exp3=use_exp3)
            
            return_vae_list.append(np.mean(ckp_return_vae_list))
            return_pi_list.append(np.mean(ckp_return_pi_list))
            return_bandit_list.append(np.mean(ckp_eturn_list_bandit))

            result_dict = {}
            result_dict['version'] = test_id
            result_dict['num_episodes'] = i_episode
            result_dict['greedy'] = return_vae_list
            result_dict['pi'] = return_pi_list
            result_dict['gscu'] = return_bandit_list
            result_dict['settings'] = settings
            
            pickle.dump(result_dict, open(rst_dir+'/online_adaption_opp_adaptive_'+test_id+'_opp'+args.opp_init_id+'.p', "wb"))

            logging.info("episode: {}, opp init id: {}, gscu: {:.2f}, | greedy: {:.2f}, | pi: {:.2f}".format(
                        i_episode,args.opp_init_id,np.mean(ckp_eturn_list_bandit),np.mean(ckp_return_vae_list),np.mean(ckp_return_pi_list)))

        if len(opponent_vae.buffer) >= batch_size:
            opponent_vae.update(i_episode)
            opponent_pi.update(i_episode)
            opponent_bandit.update(i_episode)
        
        # update vae embedding generation
        act_adv = np.array(act_adv)
        act_adv = np.argmax(act_adv.astype(np.float32), axis=1)
        obs_adv_tensor = torch.FloatTensor(obs_adv)
        act_adv_tensor = torch.FloatTensor(act_adv)
        vi.update(obs_adv_tensor, act_adv_tensor)

        act_opponent_bandit = np.array(act_opponent_bandit)
        act_opponent_bandit = np.argmax(act_opponent_bandit.astype(np.float32), axis=1)
        obs_adv_tensor_bandit = torch.FloatTensor(obs_opponent_bandit)
        act_adv_tensor_bandit = torch.FloatTensor(act_opponent_bandit)
        vi_bandit.update(obs_adv_tensor_bandit, act_adv_tensor_bandit)

        if use_exp3:
            exp3.update(episode_return_bandit, agent_selected)

    print ('avg gscu', np.mean(return_bandit_list), 
            '| avg greedy', np.mean(return_vae_list), 
            '| avg pi', np.mean(return_pi_list))
    print ('version',test_id)
    print ('seed',seed)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, help='Num of steps in a single run')
    parser.add_argument('-v', '--version', default='v0', help='version')
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-oid', '--opp_init_id', default='1', help='opponents initial weight id')    
    parser.add_argument('-e', '--encoder_file', default='encoder_vae_param_demo.pt', help='vae encoder file')
    parser.add_argument('-d', '--decoder_file', default='decoder_param_demo.pt', help='vae decoder file')
    parser.add_argument('-r', '--rl_file', default='params_demo.pt', help='conditional RL file')
    args = parser.parse_args()

    main(args)

