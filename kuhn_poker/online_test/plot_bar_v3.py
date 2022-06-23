import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from operator import add

# sns.set_theme(style="whitegrid")
sns.set_theme(style="white", palette=None,font="Verdana")

def get_mean(df,name,n_point):
    return np.mean(df[name][:n_point])

def parse_data(files,name_list,n_point=None):
    data_dict = {}
    for file in files:
        data = pd.read_pickle(file)
        for name in name_list:
            reward = get_mean(data,name,n_point)
            if name not in data_dict:
                data_dict[name] = [reward]
            else:
                data_dict[name].append(reward)
    return data_dict

def get_std(reward_list, seq_len):
    n_seq = len(reward_list)//seq_len
    print ('n_seq',n_seq)
    seq_mean = []
    for i in range(n_seq):
        seq_mean.append(np.mean(reward_list[i*seq_len:(i+1)*seq_len]))
    print ('seq_mean',seq_mean)
    std = np.std(seq_mean)
    print ('std',std)
    return std 

def cal_domain_avg(dict_list, name_list, is_seen, n_seed):
    overall_dict = {}
    avg_dict = {}
    offset_dict = {}
    domain_best_list = []
    worst_case_dict = {}

    for idx, domain_dict in enumerate(dict_list):
        domain_best = [-100000]*n_seed[idx]
        for name in name_list:
            rewards = domain_dict[name]
            if name not in overall_dict:
                overall_dict[name] = rewards
            else:
                prev_reward = overall_dict[name]
                overall_dict[name] = list(map(add, prev_reward, rewards))
            for i in range(n_seed[idx]):            
                if rewards[i] > domain_best[i]:
                    domain_best[i] = rewards[i]
        domain_best_list.append(domain_best)
    
    # avg
    for name in name_list:
        overall_dict[name] = [i/len(is_seen) for i in overall_dict[name]]
        mean_reward = np.mean(overall_dict[name])
        std_reward = np.std(overall_dict[name])
        print ('avg', name, 'mean', mean_reward, 'std',std_reward)
        avg_dict[name] = overall_dict[name]

    # worst case
    for idx, domain_dict in enumerate(dict_list):
        domain_best = domain_best_list[idx]
        for name in name_list:
            rewards = domain_dict[name]
            offset = list(np.array(rewards) - np.array(domain_best))
            if name in offset_dict:
                offset_dict[name].append(offset)
            else:
                offset_dict[name] = [offset]
    for name in name_list:
        offset = offset_dict[name]
        offset = np.array(offset)
        worst_case = np.min(offset,axis=0)
        mean_worst_case = np.mean(worst_case)
        std_worst_case = np.std(worst_case)
        print ('worst case', name, 'mean', mean_worst_case, 'std',std_worst_case)
        worst_case_dict[name] = worst_case
    return avg_dict, worst_case_dict

seen_files = ['results/online_adaption_v52_0.p','results/online_adaption_v52_1.p','results/online_adaption_v52_2.p','results/online_adaption_v52_3.p','results/online_adaption_v52_4.p']
unseen_file = ['results/online_adaption_v53_0.p','results/online_adaption_v53_1.p','results/online_adaption_v53_2.p','results/online_adaption_v53_3.p','results/online_adaption_v53_4.p']
mix_file = ['results/online_adaption_v54_0.p','results/online_adaption_v54_1.p','results/online_adaption_v54_2.p','results/online_adaption_v54_3.p','results/online_adaption_v54_4.p']
adaptive_file = [
                '../results/online_test/online_adaption_opp_adaptive_v1_1.p',
                '../results/online_test/online_adaption_opp_adaptive_v1_2.p',
                '../results/online_test/online_adaption_opp_adaptive_v1_3.p',
                '../results/online_test/online_adaption_opp_adaptive_v1_4.p',
                '../results/online_test/online_adaption_opp_adaptive_v1_5.p',
                ]


seq_len  = 20

is_overall = True
is_overall = False

# comparision_list = ['Tracking','DRON','LIAM','Deep BPR+','$\pi_1^*$','GSCU-Greedy(Ours)','GSCU(Ours)']
comparision_list = ['Tracking','DRON','LIAM','Deep BPR+','$\pi_1^*$','GSCU-Greedy(Ours)','GSCU(Ours)']
name_list = ['ne','greedy','gscu']

color_list = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', 'slategray','brown','purple','r']
is_seen = ['seen','unseen','mix','adaptive']
# is_seen = ['seen','unseen','mix']

# seen_data = pd.read_pickle(seen_file)
# unseen_data = pd.read_pickle(unseen_file)
# mix_data = pd.read_pickle(mix_file)

# seen_dict = parse_data(seen_files,name_list_prev)
# unseen_dict = parse_data(unseen_file,name_list_prev)
# mix_dict = parse_data(mix_file,name_list_prev)

adaptive_dict = parse_data(adaptive_file,name_list)

for key in adaptive_dict:
    print (key, np.mean(adaptive_dict[key]))

# dict_list = [seen_dict,unseen_dict,mix_dict,adaptive_dict]
# # avg_dict, worst_case_dict = cal_domain_avg(dict_list, name_list, is_seen, n_seed=[5,5,5,25])
# avg_dict, worst_case_dict = cal_domain_avg(dict_list, name_list, is_seen, n_seed=[5,5,5,5])



# if is_overall:
#     plt.figure(figsize=(8,6))
#     x = np.arange(2) 

#     total_width = 0.8
#     n = len(name_list)
#     width = total_width/n
#     x = x - (total_width-width)/2

#     for idx, name in enumerate(name_list):
#         mean_array = [np.mean(avg_dict[name]),np.mean(worst_case_dict[name])]
#         std_array = [np.std(avg_dict[name]),np.std(worst_case_dict[name])]
#         print (name, worst_case_dict[name])
#         plt.bar(x+width*idx, mean_array, width = width, yerr=std_array, facecolor = color_list[idx],  label=comparision_list[idx])
#         # print (name, std_array)
#     ax = plt.gca()
#     ax.xaxis.set_ticks_position('top')
#     plt.xticks(x+width*2.5, ['Average','Worst Case'], fontsize=26)
#     # plt.legend(loc="upper right") 
#     plt.legend(fontsize=16, loc="lower left") # 0612: 14 to 16
#     # plt.legend(fontsize=10, loc="upper right")
#     plt.title('Kuhn Poker', fontsize=33)
#     plt.ylabel('Returns', fontsize=22)
#     plt.yticks(fontsize=18)

#     # plt.grid(True, axis='y',ls=':',color='r',alpha=0.3) 
#     plt.show()
#     # plt.savefig('results/main_plot/kp_bar_overall_cr.pdf', bbox_inches='tight', dpi=100)
#     # plt.savefig('kp_bar_overall_cr.pdf', bbox_inches='tight', dpi=100)
#     # plt.close()
# else:
#     plt.figure(figsize=(8,6))
#     # plt.figure(figsize=(12,6))
#     x = np.arange(len(is_seen)) 

#     # total_width = 0.8
#     total_width = 0.8
#     n = len(name_list)
#     width = total_width/n
#     x = x - (total_width-width)/2

#     for idx, name in enumerate(name_list):
#         mean_array = [np.mean(seen_dict[name]),np.mean(unseen_dict[name]),np.mean(mix_dict[name]),np.mean(adaptive_dict[name])]

#         print (name, mean_array)
#         std_array = [np.std(seen_dict[name]),np.std(unseen_dict[name]),np.std(mix_dict[name]),np.std(adaptive_dict[name])]
#         if idx == 0:
#             plt.bar(x+width*idx, mean_array, width = width, yerr=std_array, facecolor = color_list[idx],  label=comparision_list[idx])
#         else:
#             plt.bar(x+width*idx, mean_array, width = width, yerr=std_array, facecolor = color_list[idx],  label=comparision_list[idx])
#         # print (name, std_array)

#     ax = plt.gca()
#     ax.xaxis.set_ticks_position('top')
#     plt.xticks(x+width*2.5, is_seen, fontsize=26)
#     # plt.legend(loc="upper right") 
#     # plt.legend(fontsize=16, loc="lower left")
#     plt.legend(fontsize=16, loc="lower left")  # 0612: 14 to 16
#     # plt.legend(fontsize=12, loc="lower left")
#     # plt.legend(fontsize=10, loc="upper right")
#     plt.title('Kuhn Poker', fontsize=33)
#     plt.ylabel('Returns', fontsize=22)
#     plt.yticks(fontsize=18)

#     # plt.grid(True, axis='y',ls=':',color='r',alpha=0.3) 
#     plt.show()
#     # plt.savefig('results/main_plot/kp_bar_cr.pdf', bbox_inches='tight', pad_inches=0.1, dpi=100)
#     # plt.savefig('kp_bar_cr.pdf', bbox_inches='tight', pad_inches=0.1, dpi=100)
#     # plt.close()
  




