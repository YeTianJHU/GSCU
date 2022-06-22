import numpy as np
import torch
import pickle
import pandas as pd 
from torch.autograd import Variable
import torch.nn.functional as F
from opponent_models import Encoder,EncoderVAE,Decoder
import logging

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from matplotlib.patches import Ellipse, Circle
from sys import platform
import seaborn as sns

obs_dim = 16
# num_adv_pool = 14
# num_adv_pool = 23
num_adv_pool = 4
action_dim = 7
hidden_dim = 128
# latent_dim = 128
latent_dim = 2
# file_name = '36_17'
# file_name = '03'

if platform == 'linux':
    is_generate = True
else:
    is_generate = False
# is_generate = True
# is_generate = False
    
def samele_and_save(file_name):
    encoder_path = '../model_params/VAE/encoder_vae_param_v'+file_name+'.pt'
    # encoder_path = 'saved_model_params/params_4000.vae.pt'
    # encoder_path = 'encoder_ae_param_03.pt'
    encoder_vae = EncoderVAE(num_adv_pool, hidden_dim, latent_dim)
    # encoder_vae = Encoder(num_adv_pool, hidden_dim, latent_dim)
    encoder_vae.load_state_dict(torch.load(encoder_path,map_location=torch.device('cpu')))
    # encoder_vae.load_state_dict(torch.load(encoder_path)['agent_params']['encoder'])
    encoder_vae.eval()


    embedding_data = {}
    for adv_id in range(num_adv_pool):
        adv_id_tensor = Variable(torch.tensor([adv_id for i in range(2000)]))
        adv_id_onehot = F.one_hot(adv_id_tensor, num_classes=num_adv_pool)
        embedding_vae,mu,logvar = encoder_vae(adv_id_onehot.float())

        embedding_vae = embedding_vae.detach().numpy()
        mu = mu.detach().numpy()
        logvar = logvar.detach().numpy()

        embedding_data[adv_id] = [embedding_vae,mu,logvar]
    pickle.dump(embedding_data, open('embedding_tag_'+file_name+'.p', "wb"))
    return embedding_data

def plot_vae(file_name):
    tsne = TSNE(n_components=2)

    n_sample = 600

    data = pd.read_pickle('embedding_tag_'+file_name+'.p')
    label_all = []
    label = []
    var_list = []
    mu_list = []
    for idx, i in enumerate(data):
        # data_i = data[i]
        
        data_i = data[i][0]
        print (data_i.shape)

        log_var = data[i][2][0,:]
        mu = data[i][1][0,:]
        var_list.append(np.exp(log_var/2))
        mu_list.append(mu)

        # print (i, mu)
        print (i, mu, np.exp(log_var))
        
        
        if idx == 0:
            data_all = data_i[:n_sample,:]
        else:
            data_all = np.vstack((data_all,data_i[:n_sample:]))
        label_all += [i for _ in range(n_sample)]
        label.append(i)

    print ('label',label)
    print ('data_all',data_all.shape)
    # X_2d_tsne = tsne.fit_transform(data_all)
    X_2d = np.array(mu_list)
    X_2d_all = np.array(data_all)

    label_all = np.array(label_all)
    label = np.array(label)
    print ('label_all',label_all.shape)


    # target_ids = [0,1,2,3,4,5,6,7,8,9]
    # target_ids = [0,1,2,3,4,5,6,7]
    target_ids = [0,1,2,3]
    
    # agent_names = ['A','B','C','D','E','F','G','Random','PPO','PPO_VAE','PPO_B','gen','v18','v37',
    #                 'maddpgv0_0','moddpgv0_1','moddpgv0_2',
    #                 'maddpgv1_0','moddpgv1_1','moddpgv1_2',
    #                 'maddpgv2_0','moddpgv2_1','moddpgv2_2']

    # agent_names = ['PolicyA', 'PolicyE', 
    #             'PolicyPPO', 'PolicyPPO_VAE', 'PolicyPPO_agb',
    #             'PolicyPPO_gen', 'PolicyPPO_v37',
    #             'maddpg_adv_v2_0','maddpg_adv_v2_1','maddpg_adv_v2_2']
    # agent_names = ['$O_N$', '$O_E$', '$O_W$', '$O_S$', '$O_{stay}$', '$O_{random}$', '$O_{rl1}$', '$O_{rl2}$']
    # agent_names = ['o$_N$', 'o$_E$', 'o$_A$', 'o$_{Stay}$']
    # agent_names = ['o$_N$', 'o$_{WN}$', 'o$_W$', 'o$_{WS}$']
    # agent_names = ['o$_N$', 'o$_E$', 'o$_W$', 'o$_A$', 'o$_{stay}$', 'o$_S$', 'o$_{rl1}$', 'o$_{rl2}$']

    # agent_names = ['o$_N$', 'o$_W$', 'o$_S$', 'o$_E$']
    agent_names = ['o$_N$', 'o$_{NW}$', 'o$_W$', 'o$_{SW}$', 'o$_{NE}$', 'o$_S$', 'o$_{SE}$', 'o$_E$']
    # agent_names = ['D','B','A','C']
    # adv_pool = ['PolicyN', 'PolicyWN', 'PolicyW', 'PolicyWS', 'PolicyEN', 'PolicyS', 'PolicyES', 'PolicyEA']


    # rgbs = ['r', 'g', 'b', 'c', 'm', 'y', 'orange','k','brown','cyan']
    # rgbs = ['orange', 'tomato', 'lime', 'g']
    # rgbs = ['orange', 'tomato', 'lime', 'g', 'brown', 'cyan', 'lightseagreen','deepskyblue']
    rgbs = ['orange', 'tomato', 'lime', 'skyblue', 'brown', 'cyan', 'lightseagreen','deepskyblue'] # CR version
    # rgbs = ['purple','r',u'#1f77b4','g'] 
    # rgbs = ['orange', 'tomato', 'lime', 'g', 'darkgreen', 'springgreen', 'lightseagreen','deepskyblue','skyblue','lightskyblue']
    # rgbs = ['r', 'g', 'b', 'c', 'm', 'y', 'orange','k','brown','cyan','lime','navy','plum','peru','skyblue','k','r']
    # rgbs = ['g','g','orange','orange','g','r','r','orange','g','y','y','y','g','g','g','g','y']
    # target_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    # agent_names = ['A','B','C','D','E','F','G','Random','PPO','PPO_VAE','PPO_B','gen','v18','v37']
    # rgbs = ['r', 'g', 'b', 'c', 'm', 'y', 'orange','k','brown','cyan','lime','tan','plum','peru']
    '''
    for i, c, l in zip(target_ids, rgbs, agent_names):
        plt.scatter(X_2d_tsne[label_all == i, 0], X_2d_tsne[label_all == i, 1], c=c, label=l, s=8)
    plt.legend()
    plt.show()
    plt.close()
    '''

    # plt.rcParams["figure.figsize"] = (8,6)
    # for a,b in zip([0],[1]):
    # # for a,b in zip([2],[7]):
    # # for a,b in zip([0,1,2,3,4,5,6],[1,2,3,4,5,6,7]):
    #     print ('axis:', a,b)
    #     fig, ax = plt.subplots()
    #     plt.rcParams["figure.figsize"] = (8,6)
    #     for i, c, l in zip(target_ids, rgbs, agent_names):

    #         plt.text(X_2d[label == i, a], X_2d[label == i, b], l, fontsize=23)
    #         # print (var_list[i])
    #         plt.scatter(X_2d[label == i, a], X_2d[label == i, b], c=c, label=l, s=8, alpha=0.5)
    #         # circle1 = Ellipse((X_2d[label == i, a], X_2d[label == i, b]), 3*np.sqrt(var_list[i][a]), 3*np.sqrt(var_list[i][b]), color=c, alpha=0.3)
    #         circle1 = Ellipse((X_2d[label == i, a], X_2d[label == i, b]), 3*var_list[i][a], 3*var_list[i][b], color=c, alpha=0.3)
    #         ax.add_patch(circle1)

    #         # print (l,'_mean',X_2d[label == i, a], X_2d[label == i, b],'std',var_list[i][a],var_list[i][b])
    #         print ('[[',X_2d[label == i, a][0],',', X_2d[label == i, b][0],'],[',var_list[i][a],',',var_list[i][b],']],')
    #     # plt.legend()
    #     # plt.rcParams["figure.figsize"] = (10,8)
    #     # plt.xlim(-4,3)
    #     # plt.ylim(-3,3)
    #     # plt.show()
    #     # plt.close()
    # # if file_name == '47_tmp_13':
    # #     plt.show()
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.savefig('vae_fig_'+file_name+'.jpg')
    # plt.close()
    
    plt.rcParams["figure.figsize"] = (8,6)
    for a,b in zip([0],[1]):
    # for a,b in zip([0,1,2,3,4,5,6],[1,2,3,4,5,6,7]):
        print ('axis:', a,b)
        fig, ax = plt.subplots()
        # plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams["figure.figsize"] = (8,6)
        for i, c, l in zip(target_ids, rgbs, agent_names):
            # print (var_list[i])
            plt.text(X_2d[label == i, a], X_2d[label == i, b], l, fontsize=23)
            plt.scatter(X_2d_all[label_all == i, a], X_2d_all[label_all == i, b], c=c, label=l, s=10, alpha=0.5)
        # plt.legend()
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # plt.rcParams["figure.figsize"] = (10,8)
        # plt.xlim(-4,3)
        # plt.ylim(-3,3)
    # print("save fig")

    # plt.savefig('vae_fig_sample'+file_name+'.pdf', bbox_inches='tight')
    plt.savefig('demo.pdf', bbox_inches='tight')
    plt.close()
    
    

file_name_list = ['0_29']

# file_name_list = ['params_4000.vae']

for file_name in file_name_list:
    if is_generate:
        _ = samele_and_save(file_name)
    else:
        plot_vae(file_name)

'''
adv_type_list = ['o$_{ES}$', 'o$_E$', 'o$_S$']
adv_file_list = ['v0', 'v1', 'v2']
color_list = ['cyan', 'lightseagreen','black']

for select_adv_idx in range(3):
    file_vi = '../bayesian_test/results/sequence_test_' + adv_file_list[select_adv_idx] + '_unseen_21.p'
    data_vi = pd.read_pickle(file_vi)
    all_mu_list = data_vi['embedding_list_gscu']
    print(all_mu_list)
    for item in all_mu_list:
        # print(item)
        plt.scatter(x=item[0], y=item[1], color=color_list[select_adv_idx], s=10)


plt.savefig('results/vae_fig_adv.pdf', bbox_inches='tight')
plt.close()
'''




