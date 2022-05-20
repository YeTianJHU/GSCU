# -*- coding:utf-8 -*-

class Config:

    OBS_DIM = 11
    NUM_ADV_POOL = 3
    ACTION_DIM = 2
    HIDDEN_DIM = 128
    LATENT_DIM = 2

    PURE_PO = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    NE_PO = [[gamma/3,(1+gamma)/3,gamma] for gamma in [0,0.2,0.4,0.6,0.8,1]]
    SAMPLE_P1 = [[0.25,0.67],[0.75,0.8],[0.67,0.4],[0.5,0.29],[0.28,0.10],[0.17,0.2],[1/3,1/3]]
    SAMPLE_P1_SEEN = [[0.25,0.67],[0.75,0.8],[0.17,0.2]]
    SAMPLE_P1_UNSEEN = [[0.67,0.4],[0.5,0.29],[0.28,0.10]]
    SAMPLE_P1_MIX = [[0.25,0.67],[0.75,0.8],[0.67,0.4],[0.5,0.29],[0.28,0.10],[0.17,0.2]]
    NE_P1 = [[1/3,1/3]]

    DATA_DIR = '../data/'
    VAE_MODEL_DIR = '../model_params/VAE/'
    VAE_RST_DIR = '../results/VAE/'

    RL_MODEL_DIR = '../model_params/RL/'
    RL_TRAINING_RST_DIR = '../results/RL/'

    ONLINE_TEST_RST_DIR = '../results/online_test/'

    OPPONENT_MODEL_DIR = '../model_params/opponent/'



