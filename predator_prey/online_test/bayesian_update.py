import numpy as np
import torch 
from torch.nn import KLDivLoss,CrossEntropyLoss
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from embedding_learning.opponent_models import OpponentModel
from scipy.special import softmax
import torch.nn.functional as F
from joblib import dump, load
import math

kl_loss = KLDivLoss(reduction='batchmean')
ce_loss = CrossEntropyLoss()

class BayesianUpdater():
    def __init__(self, opponent_model, game_steps=50):
        self.opponent_model = opponent_model
        self.num_adv_pool = opponent_model.num_adv_pool
        self.game_steps = game_steps
        self.latent_dim = opponent_model.latent_dim
        self.embedding_pool = None
        self.init_belief()

    # initial the belidf with uniform distribution
    def init_belief(self):
        belief = {}
        for i in range(self.num_adv_pool):
            belief[i] = [float(1/self.num_adv_pool)]
        self.belief = belief

    # calculate the KL-divergence between action of current adv and action by opponent model
    def get_action_kl(self,eposide_a,pred_a):
        eposide_a = F.one_hot(eposide_a.long(), num_classes=7)
        return kl_loss(eposide_a.double(),pred_a.double()).cpu().detach().numpy()
        
    def get_cross_entory(self,eposide_a,raw_output):
        eposide_a = eposide_a.type(torch.LongTensor)
        return ce_loss(raw_output,eposide_a).cpu().detach().numpy()

    def get_accuracy(self,eposide_a,pred_a):
        if len(pred_a.size()) == 2:
            pred_a = pred_a.squeeze(1)
        eposide_a = eposide_a.cpu().detach().numpy()
        pred_a = pred_a.cpu().detach().numpy()
        acc = np.mean(pred_a == eposide_a)
        return acc

# get normalized p, which is (1/eta)*p_j(pt_i) for all adv i in the opponent pool
    def cal_policy_prob(self,eposide_o,eposide_a):
        # KL between cur adv and advs in the pool
        kl_all_adv = []
        ce_all_adv = []
        acc_all_adv = []
        embedding_list = []
        for i in range(self.num_adv_pool):
            adv_index = torch.tensor([i]*self.game_steps) 
            raw_output,pred_a,embedding = self.opponent_model.inference_action(eposide_o, adv_index)
            ce = self.get_cross_entory(eposide_a,raw_output)
            acc = self.get_accuracy(eposide_a,pred_a)
            kl_all_adv.append(ce) # note here temporlly use ce
            ce_all_adv.append(ce)
            acc_all_adv.append(acc)
            embedding_list.append(embedding)

        p_prob_list = []
        for kl_d_this in kl_all_adv:
            p_prob_list.append(sum([kl_d/kl_d_this for kl_d in kl_all_adv]))

        ce_all_adv_str = [ '%.3f' % elem for elem in ce_all_adv ]
        self.embedding_pool = embedding_list
        # eta, the normalization factor
        eta_list = []
        for i in range(self.num_adv_pool):
            last_beta = self.belief[i][-1]
            p_prob = p_prob_list[i]
            eta_list.append(p_prob*last_beta)
        eta = sum(eta_list)

        # prob list
        norm_p_prob_list = [p_prob/eta for p_prob in p_prob_list]
        return norm_p_prob_list

    # uodate the belief using bayes's rule
    def update_belief(self,eposide_o,eposide_a, episode_rp=None):
        norm_p_prob_list = self.cal_policy_prob(eposide_o,eposide_a)
        norm_p_prob_list_str = [ '%.3f' % elem for elem in norm_p_prob_list ]
        for i in range(self.num_adv_pool):
            last_beta = self.belief[i][-1]
            norm_p_prob = norm_p_prob_list[i]
            self.belief[i].append(norm_p_prob*last_beta)

    # generate opponent embedding by weighted sum of the embeddings of adv in the pool
    def generate_cur_embedding(self):
        # get the embedding before the first game
        if not self.embedding_pool:
            embedding_pool = []
            for i in range(self.num_adv_pool):
                adv_index = torch.tensor([i]*self.game_steps)
                embedding = self.opponent_model.sample_mean(adv_index,is_reduce_mean=True)
                embedding_pool.append(embedding)
            self.embedding_pool = embedding_pool

        # weighted average by belief
        adv_embedding = np.zeros((self.latent_dim,))
        for i in range(self.num_adv_pool):
            beta = self.belief[i][-1]
            embedding = self.embedding_pool[i]
            adv_embedding += beta*embedding
        return adv_embedding

    def get_current_belief(self):
        cur_belief = [self.belief[i][-1] for i in self.belief]
        return cur_belief

    def get_current_ce(self,eposide_o,eposide_a):
        cur_embedding = self.generate_cur_embedding()
        cur_embedding = np.repeat([cur_embedding], self.game_steps, axis=0)
        raw_output,pred_a,_ = self.opponent_model.inference_action_by_emb(eposide_o, cur_embedding)
        ce = self.get_cross_entory(eposide_a,raw_output)
        acc = self.get_accuracy(eposide_a,pred_a)
        kl_d = self.get_action_kl(eposide_a,raw_output)
        print ('cur ce', ce, 'cur acc', acc, 'kl_d', kl_d)
        return ce,acc,kl_d

class OptimizationBayesianUpdater():
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.num_adv_pool = self.optimizer.n_vertices
        self.init_belief()

    # initial the belidf with uniform distribution
    def init_belief(self):
        belief = {}
        for i in range(self.num_adv_pool):
            belief[i] = [float(1/self.num_adv_pool)]
        self.belief = belief

    def cal_norm_factor(self,w):
        norm_factor = 0.0
        for i in range(self.num_adv_pool):
            last_beta = self.belief[i][-1]
            norm_factor += last_beta*w[i]
        return norm_factor

    def soft_belief(self):
        cur_belief = self.get_current_belief()
        cur_belief_soft = softmax(cur_belief)
        for i in range(self.num_adv_pool):
            self.belief[i][-1] = cur_belief_soft[i]

    # uodate the belief using bayes's rule
    def update_belief(self,episode_vet_act_probs,episode_modeled_act):
        w, result, w_softmax = self.optimizer.run(episode_vet_act_probs,episode_modeled_act)
        w_str = [ '%.3f' % elem for elem in w ]
        loss = result/(len(episode_vet_act_probs)*len(episode_modeled_act[0]))

        norm_factor = self.cal_norm_factor(w)
        for i in range(self.num_adv_pool):
            last_beta = self.belief[i][-1]
            norm_w = w[i]/norm_factor
            self.belief[i].append(norm_w*last_beta)

    def get_current_belief(self):
        cur_belief = [self.belief[i][-1] for i in self.belief]
        return cur_belief


class InsideHullBayesianUpdater():
    def __init__(self, kd_in_path, kd_out_path):
        self.init_belief()
        self.kd_in = load(kd_in_path) 
        self.kd_out = load(kd_out_path) 

    # initial the belidf with uniform distribution
    def init_belief(self):
        self.belief = [0.5]

    def get_inside_hull_prob(self,ce):
        val = np.array(ce).reshape(-1, 1)
        prob = np.exp(self.kd_in.score_samples(val))[0]
        return prob

    def get_outside_hull_prob(self,ce):
        val = np.array(ce).reshape(-1, 1)
        prob = np.exp(self.kd_out.score_samples(val))[0]
        return prob

    # uodate the belief using bayes's rule
    def update_belief(self,ce):
        inside_prob = self.get_inside_hull_prob(ce)
        outside_prob = self.get_outside_hull_prob(ce)

        normalizer = inside_prob * self.belief[-1] + outside_prob * (1-self.belief[-1]) + 1e-10
        self.belief.append((inside_prob/normalizer) * self.belief[-1])

    def get_current_belief(self):
        cur_belief = self.belief[-1]
        return cur_belief

    def get_cur_p(self,ce):
        inside_prob = self.get_inside_hull_prob(ce)
        outside_prob = self.get_outside_hull_prob(ce)
        
        if outside_prob*inside_prob == 0:
            prior = 1
        else:
            prior = outside_prob/(outside_prob+inside_prob) 
        prior = np.clip(prior, 0.1, 0.9)
        return prior

class VariationalInference():
    def __init__(self, opponent_model, latent_dim=2, n_update_times=20, game_steps=50):
        self.latent_dim = latent_dim
        self.n_update_times = n_update_times
        self.game_steps = game_steps
        self.opponent_model = opponent_model

        self.init_all()

    def update_prior(self):
        self.prev_mu = self.mu.clone().detach()
        self.prev_std = self.logvar.div(2).exp().clone().detach()
        self.prev_std = torch.clamp(self.prev_std, min=0.45)


    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        return mu + std*eps, std

    def update(self,obs_adv_tensor_1,act_adv_tensor_1):

        self.update_prior()

        this_recon_loss = 0.0
        for j in range(self.n_update_times):
            correct = 0.0

            z,std = self.reparametrize(self.mu, self.logvar)
            probs,pred_a,output = self.opponent_model.inference_action_by_emb(obs_adv_tensor_1,z)

            sample = torch.distributions.Normal(self.mu, std)
            prev_sample = torch.distributions.Normal(self.prev_mu, self.prev_std)
            kl_loss = torch.distributions.kl_divergence(sample, prev_sample)
            kl_loss = kl_loss.mean()
            recon_loss = ce_loss(output,act_adv_tensor_1.type(torch.LongTensor))

            loss = recon_loss + kl_loss
            self.mu.retain_grad()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            this_recon_loss += recon_loss.cpu().detach().numpy()

        ce = this_recon_loss/self.n_update_times
        self.ce_list.append(ce)

    def get_cur_ce(self):
        if len(self.ce_list) == 0:
            print ('You need to update the parameters first')
            return None 
        return self.ce_list[-1]

    def generate_cur_embedding(self,is_np=True):
        if is_np:
            mu_mean = np.mean(self.mu.cpu().detach().numpy(),axis=0)
        else:
            mu_mean = torch.mean(self.mu, axis=0)
            mu_mean = torch.reshape(mu_mean, (1, -1))
        return mu_mean

    def get_mu_and_sigma(self):
        mu_mean = np.mean(self.mu.cpu().detach().numpy(),axis=0)
        logvar_mean = np.mean(self.logvar.div(2).exp().cpu().detach().numpy(),axis=0)
        return mu_mean,logvar_mean

    def init_all(self):
        self.mu = torch.autograd.Variable(torch.tensor(self.game_steps*[[0.0 for _ in range(self.latent_dim)]]), requires_grad=True)
        self.logvar = torch.autograd.Variable(torch.tensor(self.game_steps*[[0.0 for _ in range(self.latent_dim)]]), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.mu,self.logvar], lr=0.01)
        self.ce_list = []
        self.update_prior()

    def init_var(self):
        self.logvar = torch.autograd.Variable(torch.tensor(self.game_steps*[[0.0 for _ in range(self.latent_dim)]]), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.mu,self.logvar], lr=0.01)
        self.update_prior()

class SwitchBoard():
    def __init__(self, window_size, check_size, threshold, freeze_time_threshold):
        self.window_size = window_size
        self.check_size = check_size
        self.threshold = threshold
        self.ce_list_0 = []
        self.ce_list_1 = []
        self.ce_list_2 = []
        self.freeze_time = 0
        self.freeze_time_threshold = freeze_time_threshold
        self.mean_x_list_0 = []
        self.mean_y_list_0 = []
        self.mean_x_list_1 = []
        self.mean_y_list_1 = []
        self.mean_x_list_2 = []
        self.mean_y_list_2 = []
        self.dist_change_list = []

    def cal_change_val(self, ce_list):
        change_val = np.mean(ce_list[:-self.check_size]) - np.mean(ce_list[-self.check_size:])
        return change_val

    def cal_dist_change_val(self, mu_list):
        if len(mu_list) < 2:
            return 0.05
        change_val = np.mean(mu_list[-2]) - np.mean(mu_list[-1])
        return change_val

    def update_ce(self,ces):
        self.ce_list_0.append(ces[0])
        self.ce_list_1.append(ces[1])
        self.ce_list_2.append(ces[2])
        if len(self.ce_list_0) > self.window_size:
            self.ce_list_0.pop(0)
            self.ce_list_1.pop(0)
            self.ce_list_2.pop(0)
        return self.detect_change()

    def update_mu(self,embs):
        self.mean_x_list_0.append(embs[0][0])
        self.mean_y_list_0.append(embs[0][1])
        self.mean_x_list_1.append(embs[1][0])
        self.mean_y_list_1.append(embs[1][1])
        self.mean_x_list_2.append(embs[2][0])
        self.mean_y_list_2.append(embs[2][1])

        if len(self.ce_list_0) > self.window_size:
            self.mean_x_list_0.pop(0)
            self.mean_y_list_0.pop(0)
            self.mean_x_list_1.pop(0)
            self.mean_y_list_1.pop(0)
            self.mean_x_list_2.pop(0)
            self.mean_y_list_2.pop(0)
        return self.check_mu_change()


    def detect_change(self):
        self.freeze_time += 1
        if len(self.ce_list_0) < self.window_size:
            return False
        change_val_0 = self.cal_change_val(self.ce_list_0)
        change_val_1 = self.cal_change_val(self.ce_list_1)
        change_val_2 = self.cal_change_val(self.ce_list_2)

        change_val_mean = (abs(change_val_0) + abs(change_val_1) + abs(change_val_2))/3 

        if change_val_mean > self.threshold*2 and self.freeze_time >= self.freeze_time_threshold//2:
            self.freeze_time = 0
            return True
        elif change_val_mean > self.threshold and self.freeze_time >= self.freeze_time_threshold:
            self.freeze_time = 0
            return True
        else:
            return False

    def check_mu_change(self):
        change_val_x_0 = self.cal_dist_change_val(self.mean_x_list_0)
        change_val_y_0 = self.cal_dist_change_val(self.mean_y_list_0)
        change_val_x_1 = self.cal_dist_change_val(self.mean_x_list_1)
        change_val_y_1 = self.cal_dist_change_val(self.mean_y_list_1)
        change_val_x_2 = self.cal_dist_change_val(self.mean_x_list_2)
        change_val_y_2 = self.cal_dist_change_val(self.mean_y_list_2)
        dist_delta_0 = math.sqrt(change_val_x_0**2 + change_val_y_0**2) 
        dist_delta_1 = math.sqrt(change_val_x_1**2 + change_val_y_1**2) 
        dist_delta_2 = math.sqrt(change_val_x_2**2 + change_val_y_2**2) 

        dist_delta = np.mean([dist_delta_0,dist_delta_1,dist_delta_2])

        self.dist_change_list.append(dist_delta)
        if len(self.dist_change_list) > self.window_size:
            self.dist_change_list.pop(0)
        if len(self.dist_change_list) >= self.window_size:
            change_val_dist_delta = abs(self.cal_change_val(self.dist_change_list))
        else:
            change_val_dist_delta = 0
        return dist_delta,change_val_dist_delta


class EXP3():
    def __init__(self, n_action, gamma, min_reward, max_reward):
        self.n_action = n_action
        self.gamma = gamma
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.init_weight()

    def init_weight(self):
        self.w = [1.0] * self.n_action
        self.prior = [0.5,0.5]
        self.generate_p()

    def generate_p(self):
        weights = self.w
        weights = np.multiply(weights, self.prior)
        gamma = self.gamma
        weight_sum = float(sum(weights))
        p = [(1.0 - gamma) * (w / weight_sum) + (gamma / len(weights)) for w in weights]
        self.p = p

    def sample_action(self):
        self.generate_p()
        rand = np.random.random() 
        for i in range(self.n_action):
            rand -= self.p[i]
            if rand <= 0:
                return i
        return i

    def scale_reward(self, reward):
        if reward > self.max_reward:
            return 1
        if reward < self.min_reward:
            return 0
        return (reward - self.min_reward) / (self.max_reward - self.min_reward)

    def update(self, reward, action_idx):
        scaled_reward = self.scale_reward(reward)
        estimated_reward  = 1.0*scaled_reward/self.p[action_idx]

        self.w[action_idx] *= math.exp(estimated_reward * self.gamma / self.n_action)
        if self.w[0] > 1e10 or self.w[1] > 1e10:
            self.w[0] *= 1e-5
            self.w[1] *= 1e-5
                 
    def get_p(self):
        self.generate_p()
        if np.isnan(self.p[0]) or np.isnan(self.p[1]):
            print ('nan', self.w)
            self.init_weight()
        return self.p

    def add_prior(self, prior):
        self.prior = [1-prior,prior]


if __name__ == '__main__':
    obs_dim = 16 # observation dimension
    num_adv_pool = 4 # policies pool size
    action_dim = 7 
    hidden_dim = 128
    latent_dim = 2
    game_steps = 50
    encoder_weight_path = '../model_params/VAE/encoder_vae_param_demo.pt'
    decoder_weight_path = '../model_params/VAE/decoder_param_demo.pt'

    opponent_model = OpponentModel(obs_dim, num_adv_pool, hidden_dim, latent_dim, 
                action_dim, encoder_weight_path, decoder_weight_path)

    eposide_o = torch.autograd.Variable(torch.rand(game_steps, obs_dim)) 
    eposide_a = torch.autograd.Variable(torch.randint(low=0, high=action_dim, size=(game_steps,))) 

    vi1 = VariationalInference(opponent_model, latent_dim=latent_dim, n_update_times=50, game_steps=game_steps)
    vi1.init_all()
    vi1.update(eposide_o,eposide_a)
    cur_ce_1 = vi1.get_cur_ce()
    cur_latent1_np = vi1.generate_cur_embedding(is_np=True)
    print ('ce loss', cur_ce_1)
    print ('latent emb', cur_latent1_np)

    exp3 = EXP3(n_action=2, gamma=0.2, min_reward=-500, max_reward=100)
    exp3.init_weight()
    agent_selected = exp3.sample_action()
    print ('agent_selected',agent_selected)
    exp3.update(-100,agent_selected)
    agent_selected = exp3.sample_action()
    print ('agent_selected',agent_selected)
    exp3.update(-60,agent_selected)
    p = exp3.get_p()
    print ('action select prob',p)

    embs = [np.array([-0.01765536,  0.2418145 ]), np.array([-0.05306276,  0.06705052]), np.array([ 0.06372164, -0.06746028])]
    ces_0 = [0.1,0.5,0.2]
    ces_1 = [1.1,0.9,0.8]
    swith_board = SwitchBoard(window_size=50, check_size=10, threshold=0.15, freeze_time_threshold=70)
    is_change = swith_board.update_ce(ces_0)
    is_change = swith_board.update_ce(ces_1)
    dist_delta = swith_board.update_mu(embs)
    print ('is_change',is_change)
    print ('is_change',is_change)
    print ('dist_delta',dist_delta)




