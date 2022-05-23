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
import pickle
import logging
import random
import pyspiel
import pandas as pd 
import glob 
from embedding_learning.opponent_models import *
from online_test.bayesian_update import VariationalInference,EXP3
from conditioned_RL.conditional_rl_model import PPO_VAE
from utils.config_kuhn_poker import Config
from utils.mypolicy import PolicyKuhn,get_policy_by_vector,BestResponseKuhn
from utils.utils import get_p1_region,get_onehot,kl_by_mean_sigma,mse

np.set_printoptions(precision=6)
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def print(*args):
    __builtins__.print(*("%.4f" % a if isinstance(a, float) else a
                         for a in args))
def moving_avg(x,N=5):
    return np.convolve(x, np.ones(N)/N, mode='same')

def region2index(region):
    if region == 7:
        return 0
    elif region == 3:
        return 1
    elif region == 5:
        return 2 
    else:
        return -1

def evaluate_exp3(n_test, player, agent_vae, opponent_policy, opponent_type, latent, ne_response, exp3):
    game = pyspiel.load_game("kuhn_poker(players=2)") 
    return_list = []
    for _ in range(n_test):
        state = game.new_initial_state()
        agent_selected = exp3.sample_action()
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
                    if agent_selected == 0:
                        _, action, _ = agent_vae.select_action(s, latent)
                    else:
                        action = ne_response.action(s)
                else:
                    if opponent_type == 'rl':
                        _, action, _ = opponent_policy.select_action(s)
                    else:
                        action = opponent_policy.action(s)
                state.apply_action(action)
        returns = state.returns()
        this_returns = returns[player]
        return_list.append(this_returns)
    return np.mean(return_list),return_list

def evaluate_vae(n_test, player, agent_vae, opponent_policy, opponent_type, latent):
    game = pyspiel.load_game("kuhn_poker(players=2)")
    return_list = []
    for _ in range(n_test):
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
                    _, action, _ = agent_vae.select_action(s, latent)
                else:
                    if opponent_type == 'rl':
                        _, action, _ = opponent_policy.select_action(s)
                    else:
                        action = opponent_policy.action(s)
                state.apply_action(action)
        returns = state.returns()
        this_returns = returns[player]
        return_list.append(this_returns)
    return np.mean(return_list),return_list

def evaluate_baseline(n_test, player, response, opponent_policy, opponent_type):
    game = pyspiel.load_game("kuhn_poker(players=2)")
    return_list = []
    for _ in range(n_test):
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
                    action = response.action(s)
                else:
                    if opponent_type == 'rl':
                        _, action, _ = opponent_policy.select_action(s)
                    else:
                        action = opponent_policy.action(s)
                state.apply_action(action)
        returns = state.returns()
        this_returns = returns[player]
        return_list.append(this_returns)
    return np.mean(return_list),return_list


def main(args):

    state_dim = Config.OBS_DIM
    action_dim = Config.ACTION_DIM
    n_adv_pool = Config.NUM_ADV_POOL
    embedding_dim = Config.LATENT_DIM
    hidden_dim = Config.HIDDEN_DIM

    actor_lr = 5e-4  
    critic_lr = 5e-4 

    n_steps = 10 # vi update freq
    this_player = 0 # controlling player
    n_opponent = 200 # total number of opponent switch (we tested 10 sequences with 20 opponents each)
    reset_freq = 20 # sequence length
    n_episode = 1000 # number of episodes per opponent switch
    evaluation_freq = n_steps * 15 # evaluation freq
    n_test = 200 # number of evaluation epsideos. More than 1000 episodes are recommanded.  

    version = args.version 
    opponent_type = args.opp_type
    print ('opponent_type',opponent_type)

    seed = int(args.seed)
    version += '_' + str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    encoder_weight_path = Config.VAE_MODEL_DIR + args.encoder_file
    decoder_weight_path = Config.VAE_MODEL_DIR + args.decoder_file
    conditional_rl_weight_path = Config.RL_MODEL_DIR + args.rl_file
    opponent_model = OpponentModel(state_dim, n_adv_pool, hidden_dim, embedding_dim, action_dim, encoder_weight_path, decoder_weight_path)
    vi = VariationalInference(opponent_model, latent_dim=embedding_dim, n_update_times=50, game_steps=n_steps)
    exp3 = EXP3(n_action=2, gamma=0.3, min_reward=-2, max_reward=2) # lr of exp3 is set to 0.3
    agent_vae = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, n_adv_pool)
    agent_vae.init_from_save(conditional_rl_weight_path)

    rst_dir = Config.ONLINE_TEST_RST_DIR
    data_dir = Config.DATA_DIR

    # a randomly generated sequence is used. You can create your own.
    policy_vectors_df = pd.read_pickle(data_dir+'online_test_policy_vectors_demo.p')
    if opponent_type == 'seen':
        policy_vectors = policy_vectors_df['seen']
    elif opponent_type == 'unseen':
        policy_vectors = policy_vectors_df['unseen']
    elif opponent_type == 'mix':
        policy_vectors = policy_vectors_df['mix']
    else:
        raise ValueError('No such opponent type')

    game = pyspiel.load_game("kuhn_poker(players=2)")
    state = game.new_initial_state()

    global_return_vae_list = []
    global_return_ne_list = []
    global_return_exp3_list = []

    policy_vec_list = []
    opponent_list = []

    for n in range(n_opponent):
        obs_list = []
        act_index_list = []
        return_list = []

        policy_vec = policy_vectors[n][0]
        player = this_player

        region = get_p1_region(policy_vec[3:])
        policy_index = region2index(region)
        opponent_policy = get_policy_by_vector(policy_vec,is_best_response=False)
        # NE policy
        ne_response = PolicyKuhn(0,1/3,0,1/3,1/3)

        return_vae_list = []
        return_ne_list = []
        return_best_list = []
        return_exp3_list = []

        policy_vec_list.append(policy_vec)
        opponent_list.append(region)

        # reset everything every reset_freq=20 opponent - same as change to a new sequence
        if n%reset_freq == 0:
            exp3.init_weight()
            vi.init_all() 

        for j in range(n_episode):

            latent = vi.generate_cur_embedding(is_np=False)
            state = game.new_initial_state()
            agent_selected = exp3.sample_action()

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
                        if agent_selected == 0:
                            act_vae, act_index_vae, act_prob_vae = agent_vae.select_action(s, latent)
                            action = act_index_vae
                        else:
                            action = ne_response.action(s)
                    else:
                        action = opponent_policy.action(s)
                        obs_list.append(s)
                        act_index_list.append(action)
                    state.apply_action(action)
            returns = state.returns()
            this_returns = returns[player]
            return_list.append(this_returns)

            exp3.update(this_returns,agent_selected)

            # vi update using the online data (paper version)
            # a replay buffer can also be used to boost the performnace
            if len(obs_list) >= n_steps:
                act_index = np.array(act_index_list).astype(np.float32)
                obs_adv_tensor = torch.FloatTensor(np.array(obs_list[:n_steps]))
                act_adv_tensor = torch.FloatTensor(np.array(act_index[:n_steps]))
                vi.update(obs_adv_tensor, act_adv_tensor)
                emb = vi.generate_cur_embedding(is_np=True)
                ce = vi.get_cur_ce()
                obs_list = []
                act_index_list = []
                return_list = []  

            if j%(evaluation_freq) == 0 and j > 0:
                emb,sig = vi.get_mu_and_sigma()
                emb_tensor = vi.generate_cur_embedding(is_np=False)
                ce = vi.get_cur_ce()
                p = exp3.get_p()

                avg_return_vae,return_vae = evaluate_vae(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor)
                avg_return_ne,return_ne = evaluate_baseline(n_test, player, ne_response, opponent_policy, 'rule_based')
                avg_return_exp3,return_exp3 = evaluate_exp3(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor,ne_response,exp3)

                return_vae_list.append(avg_return_vae)
                return_ne_list.append(avg_return_ne)
                return_exp3_list.append(avg_return_exp3)
                global_return_vae_list.append(avg_return_vae)
                global_return_ne_list.append(avg_return_ne)
                global_return_exp3_list.append(avg_return_exp3)

        seq_idx = n//reset_freq
        opp_idx = n%reset_freq
        logging.info("seq idx: {}, opp idx: {}, opp name: o{}, gscu: {:.2f}, | greedy: {:.2f}, | ne: {:.2f}".format(
                    seq_idx,opp_idx,region,np.mean(return_exp3_list),np.mean(return_vae_list),np.mean(return_ne_list)))

        if n%reset_freq == (reset_freq-1):
            print ('# seq: ', seq_idx, ', total # of opp: ', n,
                    ', avg gscu', np.mean(global_return_exp3_list), 
                    '| avg greedy', np.mean(global_return_vae_list), 
                    '| avg ne', np.mean(global_return_ne_list))
            print ('-'*10)

        result = {
            'opponent_type':opponent_type,
            'gscu': global_return_exp3_list,
            'greedy': global_return_vae_list,
            'ne': global_return_ne_list,
            'n_opponent':n+1,
            'policy_vec_list':policy_vec_list,
            'opponent_list':opponent_list}

        pickle.dump(result, open(rst_dir+'/online_adaption_'+version+'.p', "wb"))

    print ('version',version)
    print ('opponent_type',opponent_type)
    print ('seed',seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--version', default='v0', help='version')
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-o', '--opp_type', default='seen', 
                        choices=["seen", "unseen", "mix"], help='type of the opponents')
    parser.add_argument('-e', '--encoder_file', default='encoder_vae_param_demo.pt', help='vae encoder file')
    parser.add_argument('-d', '--decoder_file', default='decoder_param_demo.pt', help='vae decoder file')
    parser.add_argument('-r', '--rl_file', default='params_demo.pt', help='conditional RL file')
    args = parser.parse_args()
    main(args)
