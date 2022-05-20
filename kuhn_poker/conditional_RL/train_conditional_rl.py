import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
from itertools import count
import copy
import logging
import pickle
import random
import pyspiel
from embedding_learning.opponent_models import *
from conditioned_RL.PPO_VAE_gpu import PPO_VAE
from utils.config_kuhn_poker import Config
from utils.mypolicy import PolicyKuhn,get_policy_by_vector,BestResponseKuhn
from utils.utils import sample_fixed_vector,get_onehot

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

N_ADV = 1

# evalue the model with opponents sampled from the seen pool
def evaluate_training_model(n_test, player, agent_vae, config, n_adv_pool,device):
    game = pyspiel.load_game("kuhn_poker(players=2)")
    return_list = []
    for _ in range(n_test):
        policy_vec = sample_fixed_vector(config, is_seen=True,is_mix=False)
        opponent_policy = get_policy_by_vector(policy_vec[0],is_best_response=False)
        vae_vector = get_onehot(n_adv_pool,policy_vec[1])

        policy_vec_tensor = torch.tensor(np.array([vae_vector])).float().to(device)
        latent,mu,_ = agent_vae.encoder(policy_vec_tensor)

        state = game.new_initial_state()
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            cur_player = state.current_player()
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                s = state.information_state_tensor(cur_player)
                if cur_player == player:
                    act_vae, action, act_prob_vae = agent_vae.select_action(s, latent)
                else:
                    action = opponent_policy.action(s)
                state.apply_action(action)
        returns = state.returns()
        this_returns = returns[player]
        return_list.append(this_returns)
    return np.mean(return_list),return_list


def main(args):

    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'returns', 'latent'])

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    version = args.version

    # policy parameters of opponents from the seen pool
    sample_p1 = Config.SAMPLE_P1_SEEN

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gamma = 0.98
    actor_lr = 5e-4 
    critic_lr = 5e-4
    num_episodes = 200000 
    checkpoint_freq = num_episodes//10
    n_test = 5000
    evaluate_freq = num_episodes//20

    use_latent = True  # ture if using embedding/mean, else using onehot adv ids
    use_mean = False # true if using mean from encoder, instade of the embedding
    is_controls = False
    batch_size = 1024
    ppo_update_time = 5
    this_player = 0 # controling player 0

    if not use_latent:
        use_mean = False
    if is_controls:
        use_latent = True

    state_dim = Config.OBS_DIM
    action_dim = Config.ACTION_DIM
    n_adv_pool = Config.NUM_ADV_POOL
    embedding_dim = Config.LATENT_DIM
    hidden_dim = Config.HIDDEN_DIM

    vae_model_dir = Config.VAE_MODEL_DIR
    rl_training_rst_dir = Config.RL_TRAINING_RST_DIR
    rl_model_dir = Config.RL_MODEL_DIR

    encoder_weight_path = vae_model_dir + args.encoder_file
    agent_VAE = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, n_adv_pool, rl_model_dir)
    agent_VAE.batch_size = batch_size
    agent_VAE.ppo_update_time = ppo_update_time


    game = pyspiel.load_game("kuhn_poker(players=2)")
    global_evaluate_return_list = []

    n_samples = 0
    for i in range(50):
        for i_episode in range(int(num_episodes/50)):
            player = this_player

            # randomly sample opponent from the seen pool
            rand_int = np.random.randint(len(sample_p1))
            policy_vec = [0,1/3,0] + sample_p1[rand_int]
            vae_vector = get_onehot(n_adv_pool,rand_int)
            opponent_policy = get_policy_by_vector(policy_vec,is_best_response=False)

            if is_controls: # emb w/ no meaning
                latent = torch.tensor(np.array([np.zeros(embedding_dim)])).float().to(device)
            elif use_latent: # using emb
                policy_vec_tensor = torch.tensor(np.array([vae_vector])).float().to(device)
                latent,mu,_ = agent_VAE.encoder(policy_vec_tensor)
                if use_mean: # using mean
                    latent = mu
            else: # using policy_vec
                latent = torch.tensor(np.array([policy_vec[3:]])).float().to(device)
            latent_np = latent[0].cpu().detach().numpy()

            state = game.new_initial_state()
            obs_list = []
            act_index_list = []
            act_prob_list = []
            dron_feature_list = []
            while not state.is_terminal():
                legal_actions = state.legal_actions()
                cur_player = state.current_player()
                if state.is_chance_node():
                    outcomes_with_probs = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes_with_probs)
                    action = np.random.choice(action_list, p=prob_list)
                    state.apply_action(action)
                else:
                    s = state.information_state_tensor(cur_player)
                    if cur_player == player:
                        act_vae, action, act_prob_vae = agent_VAE.select_action(s, latent)
                        obs_list.append(s)
                        act_index_list.append(action)
                        act_prob_list.append(act_prob_vae)
                    else:
                        action = opponent_policy.action(s)
                        liam_op_s = s
                        liam_op_act = action
                    state.apply_action(action)
            returns = state.returns()
            # calculate reward
            this_returns = returns[player]

            n_steps = len(act_index_list)
            if n_steps == 1:
                return_list = [this_returns]
            elif n_steps == 2:
                return_list = [gamma*this_returns, this_returns] # discouted future reward

            # store 'state', 'action', 'a_log_prob', 'returns', 'latent'
            for n in range(n_steps):                 
                trans = Transition(obs_list[n], act_index_list[n], act_prob_list[n], return_list[n], latent_np)
                agent_VAE.store_transition(trans)  

            if len(agent_VAE.buffer) >= agent_VAE.batch_size:
                agent_VAE.update()

            current_episode = int(num_episodes/50 * i + i_episode)

            if current_episode % checkpoint_freq == 0:
                agent_VAE.save_params(version +'_'+str(current_episode)) 

            if current_episode % evaluate_freq == 0:
                evaluate_return = evaluate_training_model(n_test, player, agent_VAE, Config, n_adv_pool,device)
                logging.info("Average returns is {0} at the end of epoch {1}".format(evaluate_return[0], current_episode))
                global_evaluate_return_list.append(evaluate_return[0])

        result_dict = {}
        result_dict['version'] = version
        result_dict['evaluate_return_list'] = global_evaluate_return_list
        result_dict['evaluate_freq'] = evaluate_freq
        pickle.dump(result_dict, open(rl_training_rst_dir+'return_'+version+'.p', "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-v', '--version', default='v0')
    parser.add_argument('-e', '--encoder_file', default='encoder_vae_param_demo.pt', help='file name of the encoder parameters')
    args = parser.parse_args()

    main(args)

                




