# -*- coding:utf-8 -*-

class Config:

    HIDDEN_DIM = 128
    LATENT_DIM = 2
    WINDOW_SIZW = 8

    ADV_POOL_SEEN = ['PolicyN', 'PolicyNW', 'PolicyW', 'PolicySW']
    ADV_POOL_UNSEEN = ['PolicyNE','PolicySE', 'PolicyE', 'PolicyS']
    ADV_POOL_MIX = ADV_POOL_SEEN + ADV_POOL_UNSEEN

    DATA_DIR = '../data/'
    VAE_MODEL_DIR = '../model_params/VAE/'
    VAE_RST_DIR = '../results/VAE/'

    RL_MODEL_DIR = '../model_params/RL/'
    RL_TRAINING_RST_DIR = '../results/RL/'

    ONLINE_TEST_RST_DIR = '../results/online_test/'

    OPPONENT_MODEL_DIR = '../model_params/opponent/'



