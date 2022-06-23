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
import math
import glob 
from embedding_learning.opponent_models import *
from online_test.bayesian_update import VariationalInference,EXP3
from online_adaption import evaluate_exp3,evaluate_vae,evaluate_baseline
from conditional_RL.conditional_rl_model import PPO_VAE
from conditional_RL.ppo_model import PPO
from utils.config_kuhn_poker import Config
from utils.mypolicy import PolicyKuhn,get_policy_by_vector

np.set_printoptions(precision=4)

def train_opponent(game,opponent_p1,agent,agent_type,Transition_p1,latent=None):
    obs_list = []
    act_index_list = []
    act_prob_list = []
    player = 1
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
            if cur_player == 0:
                if agent_type == 'rl':
                    act_vae, action, act_prob_vae = agent.select_action(s,latent)
                else:
                    action = agent.action(s)
            else:
                act_vae, act_index_vae, act_prob_vae = opponent_p1.select_action(s)
                action = act_index_vae
                obs_list.append(s)
                act_index_list.append(act_index_vae)
                act_prob_list.append(act_prob_vae)
            state.apply_action(action)
    returns = state.returns()
    this_returns = returns[player]
    n_steps = len(act_index_list)
    if n_steps == 1:
        return_list = [this_returns]
    elif n_steps == 2:
        return_list = [gamma*this_returns, this_returns]
    for n in range(n_steps):
        trans = Transition_p1(obs_list[n], act_index_list[n], act_prob_list[n], return_list[n])
        opponent_p1.store_transition(trans)                    
    if len(opponent_p1.buffer) >= opponent_p1.batch_size:
        opponent_p1.update()



def main(args):

    state_dim = Config.OBS_DIM
    action_dim = Config.ACTION_DIM
    n_adv_pool = Config.NUM_ADV_POOL
    embedding_dim = Config.LATENT_DIM
    hidden_dim = Config.HIDDEN_DIM

    # opponent lr
    actor_lr = 5e-5 
    critic_lr = 5e-5
    # opponent batch size
    batch_size = 100 
    # discount factor for opponent learning
    gamma = 0.99

    n_steps = 10 # vi update freq
    this_player = 0 # controling player 0
    n_episode = 1000*5 
    n_test = 10000 # number of evaluation epsideos. More than 1000 episodes are recommanded.  

    version = args.version 

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    encoder_weight_path = Config.VAE_MODEL_DIR + args.encoder_file
    decoder_weight_path = Config.VAE_MODEL_DIR + args.decoder_file
    conditional_rl_weight_path = Config.RL_MODEL_DIR + args.rl_file
    opponent_p1_weight_path = Config.OPPONENT_MODEL_DIR + 'params_opp_init_' + args.opp_init_id + '.pt'

    opponent_model = OpponentModel(state_dim, n_adv_pool, hidden_dim, embedding_dim, action_dim, encoder_weight_path, decoder_weight_path)
    vi = VariationalInference(opponent_model, latent_dim=embedding_dim, game_steps=n_steps)
    exp3 = EXP3(n_action=2, gamma=0.2, min_reward=-4, max_reward=4)
    agent_vae = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, 0.00, 0.00, encoder_weight_path, n_adv_pool)
    agent_vae.init_from_save(conditional_rl_weight_path)

    opponent_p1_exp3 = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr)
    opponent_p1_exp3.init_from_save(opponent_p1_weight_path)
    opponent_p1_exp3.batch_size = batch_size
    opponent_p1_vae = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr)
    opponent_p1_vae.init_from_save(opponent_p1_weight_path)
    opponent_p1_vae.batch_size = batch_size
    opponent_p1_ne = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr)
    opponent_p1_ne.init_from_save(opponent_p1_weight_path)
    opponent_p1_ne.batch_size = batch_size

    rst_dir = Config.ONLINE_TEST_RST_DIR
    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir, exist_ok=False) 

    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'returns'])
    Transition_p1 = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'returns'])

    game = pyspiel.load_game("kuhn_poker(players=2)")

    state = game.new_initial_state()

    obs_list = []
    act_index_list = []

    ne_response = PolicyKuhn(1/3,2/3,1,1/3,1/3)
    exp3.init_weight()
    agent_selected = 1
    player = this_player

    return_vae_list = []
    return_ne_list = []
    return_exp3_list = []
    ce_list = []

    for j in range(n_episode):

        latent = vi.generate_cur_embedding(is_np=False)
    
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
                    if agent_selected == 0:
                        act_vae, act_index_vae, act_prob_vae = agent_vae.select_action(s, latent)
                        action = act_index_vae
                    else:
                        action = ne_response.action(s)
                else:
                    act_vae, action, act_prob_vae = opponent_p1_exp3.select_action(s)
                    obs_list.append(s)
                    act_index_list.append(action)
                state.apply_action(action)
        returns = state.returns()
        this_returns = returns[player]

        exp3.update(this_returns,agent_selected)
        agent_selected = exp3.sample_action()

        if len(obs_list) >= n_steps:
            act_index = np.array(act_index_list).astype(np.float32)
            obs_adv_tensor = torch.FloatTensor(np.array(obs_list[:n_steps]))
            act_adv_tensor = torch.FloatTensor(np.array(act_index[:n_steps]))
            vi.update(obs_adv_tensor, act_adv_tensor)
            emb = vi.generate_cur_embedding(is_np=True)
            obs_list = []
            act_index_list = []
            ce = vi.get_cur_ce()
            ce_list.append(ce)


        # train opponent
        emb_tensor = vi.generate_cur_embedding(is_np=False)
        if agent_selected == 0:
            train_opponent(game,opponent_p1_exp3,agent_vae,'rl',Transition_p1,latent=emb_tensor)
        else:
            train_opponent(game,opponent_p1_exp3,ne_response,'rule_based',Transition_p1)
        train_opponent(game,opponent_p1_vae,agent_vae,'rl',Transition_p1,latent=emb_tensor)
        train_opponent(game,opponent_p1_ne,ne_response,'rule_based',Transition_p1)

        if j%(n_steps*50) == 0 and j > 0:
            emb = vi.generate_cur_embedding(is_np=True)
            emb_tensor = vi.generate_cur_embedding(is_np=False)
            ce = vi.get_cur_ce()
            p = exp3.get_p()

            avg_return_vae,return_vae = evaluate_vae(n_test, player, agent_vae, opponent_p1_vae, 'rl', emb_tensor)
            avg_return_ne,return_ne = evaluate_baseline(n_test, player, ne_response, opponent_p1_ne,'rl')
            avg_return_exp3,return_exp3 = evaluate_exp3(n_test, player, agent_vae, opponent_p1_exp3, 'rl', emb_tensor,ne_response,exp3)

            logging.info("episode: {}, opp init id: {}, gscu: {:.2f}, | greedy: {:.2f}, | ne: {:.2f}".format(
                        j,args.opp_init_id,np.mean(avg_return_exp3),np.mean(avg_return_vae),np.mean(avg_return_ne)))

            return_vae_list.append(avg_return_vae)
            return_ne_list.append(avg_return_ne)
            return_exp3_list.append(avg_return_exp3)


    print ('avg gscu', np.mean(return_exp3_list), 
            '| avg greedy', np.mean(return_vae_list), 
            '| avg ne', np.mean(return_ne_list))

    result = {
        'gscu': return_exp3_list,
        'greedy': return_vae_list,
        'ne': return_ne_list}

    pickle.dump(result, open(rst_dir+'online_adaption_opp_adaptive_'+version+'_'+args.opp_init_id+'.p', "wb"))

    print ('version',version)
    print ('seed',seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--version', default='v0', help='version')
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-oid', '--opp_init_id', default='1', help='opponents initial weight id')    
    parser.add_argument('-e', '--encoder_file', default='encoder_vae_param_demo.pt', help='vae encoder file')
    parser.add_argument('-d', '--decoder_file', default='decoder_param_demo.pt', help='vae decoder file')
    parser.add_argument('-r', '--rl_file', default='params_demo.pt', help='conditional RL file')
    args = parser.parse_args()
    main(args)