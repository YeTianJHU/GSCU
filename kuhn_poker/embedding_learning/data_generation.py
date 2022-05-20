import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import random
import time 
import pickle 
import argparse
import numpy as np
import pyspiel
from utils.config_kuhn_poker import Config
from utils.mypolicy import PolicyKuhn,get_policy_by_vector
from utils.utils import get_onehot


N_TRAINING_SAMPLES = 100000
N_TESTING_SAMPLES = N_TRAINING_SAMPLES//10
SEED_TRAINING = 0
SEED_TESTING = 1
data_dir = Config.DATA_DIR
sample_p1 = Config.SAMPLE_P1_SEEN


def main(version):

    testing_mode = [True, False]
    for is_test in testing_mode:
        if is_test:
            this_version = version + '_test'
            n_sample = N_TESTING_SAMPLES
            seed = SEED_TESTING
            print ('Generating testing data...')
        else:
            this_version = version
            n_sample = N_TRAINING_SAMPLES
            seed = SEED_TRAINING
            print ('Generating training data...')

        np.random.seed(seed)
        random.seed(seed)

        game = pyspiel.load_game("kuhn_poker(players=2)")

        data_s = []
        data_a = []
        data_i = []
        data_p = []

        start_time = time.time()
        for i in range(n_sample):
            state = game.new_initial_state()

            # p0 is with random policy parameters
            policy_vector_a = np.random.rand(5) 
            policy_a = get_policy_by_vector(policy_vector_a,is_best_response=False)

            # p1 is randomly sampled from the seen p1 pool
            n_class = len(sample_p1)
            rand_int = np.random.randint(n_class)
            policy_vector_b = [0,1/3,0] + sample_p1[rand_int]
            p1_idx_onehot = get_onehot(n_class, rand_int)
            policy_b = get_policy_by_vector(policy_vector_b,is_best_response=False)

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
                    if cur_player == 0: # p0
                        action = policy_a.action(s)
                        this_policy_vec = policy_vector_a
                    else: # p1
                        action = policy_b.action(s)
                        this_policy_vec = p1_idx_onehot
                    state.apply_action(action)
                    data_s.append(s)
                    data_a.append(action)
                    data_i.append(this_policy_vec)
                    data_p.append(cur_player)
            returns = state.returns()

        end_time = time.time()
        print ('Time dur: {:.2f}s'.format(end_time-start_time))
        vae_data = {
            'data_s': data_s,
            'data_a': data_a,
            'data_i': data_i,
            'data_p': data_p}

        pickle.dump(vae_data, 
            open(data_dir+'/vae_data_kuhn_poker_'+this_version+'.p', "wb"))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--version', default='v0')
    args = parser.parse_args()

    main(args.version)

