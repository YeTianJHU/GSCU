import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import copy
import pickle
import random
import time
import collections
from utils.config_kuhn_poker import Config

def sample_fixed_vector(config, is_seen=True,is_mix=False):
    if is_mix:
        sample_p1 = config.SAMPLE_P1_MIX
    else:
        if is_seen:
           sample_p1 = config.SAMPLE_P1_SEEN
        else:
            sample_p1 = config.SAMPLE_P1_UNSEEN
    n_adv_pool = len(sample_p1)
    rand_int = np.random.randint(n_adv_pool)
    policy_vec = [0,1/3,0] + sample_p1[rand_int]
    return [policy_vec,rand_int]

def get_p1_region(p1_vec):
    eta = p1_vec[0]
    xi = p1_vec[1] 
    if eta <= 1/3 and xi <= 1/3:
        if eta >= xi:
            return 6
        else:
            return 5
    elif eta > 1/3 and xi > 1/3:
        if eta >= xi:
            return 4
        else:
            return 3
    elif eta >= xi:
        return 2
    else:
        return 7

def get_onehot(n_class,arr):
    return np.eye(n_class)[arr]

def kl_by_mean_sigma(mean1,mean2,std1,std2):
    kld_mv = 0.0
    for i in range(2):
        kld_mv += np.log(std2[i]/std1[i]) + (std1[i]**2 + (mean1[i] - mean2[i])**2)/(2*std2[i]**2) - 0.5
    kld_mv = kld_mv/2

    return kld_mv

def mse(A,B):
    mse = (np.square(A - B)).mean(axis=0)
    return mse


