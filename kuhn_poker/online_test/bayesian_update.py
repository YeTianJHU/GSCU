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
ce_loss = CrossEntropyLoss(reduction='sum')

class VariationalInference():
    def __init__(self, opponent_model, latent_dim=2, n_update_times=10, game_steps=50):
        self.latent_dim = latent_dim
        self.n_update_times = n_update_times
        self.game_steps = game_steps
        self.opponent_model = opponent_model
        self.init_all()

    def update_prior(self):
        self.prev_mu = self.mu.clone().detach()
        self.prev_std = self.logvar.div(2).exp().clone().detach()
        # prior restriction
        self.prev_std = torch.clamp(self.prev_std, min=0.5)

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
            loss = recon_loss + 1.0*kl_loss
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
        logvar_mean = np.mean(self.logvar.cpu().detach().numpy(),axis=0)
        return mu_mean,logvar_mean

    def init_all(self):
        self.mu = torch.autograd.Variable(torch.tensor(self.game_steps*[[0.0 for _ in range(self.latent_dim)]]), requires_grad=True)
        self.logvar = torch.autograd.Variable(torch.tensor(self.game_steps*[[0.0 for _ in range(self.latent_dim)]]), requires_grad=True)
        # self.optimizer = torch.optim.Adam([self.mu,self.logvar], lr=0.01)
        self.optimizer = torch.optim.Adam([self.mu,self.logvar], lr=0.005)
        self.ce_list = []
        self.update_prior()

###
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
        # weights = np.multiply(weights, self.prior)
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
    num_adv_pool = 11 # policies pool size
    action_dim = 7 
    hidden_dim = 128
    latent_dim = 8
    game_steps = 50
    encoder_weight_path = 'saved_model_params/encoder_vae_param_v35_9.pt'
    decoder_weight_path = 'saved_model_params/decoder_param_v35_9.pt'
    # encoder_weight_path = 'saved_model_params/encoder_vae_param_v26_4.pt'
    # decoder_weight_path = 'saved_model_params/decoder_param_v26_4.pt'

    opponent_model = OpponentModel(obs_dim, num_adv_pool, hidden_dim, latent_dim, 
                action_dim, encoder_weight_path, decoder_weight_path)


    eposide_o = torch.autograd.Variable(torch.rand(game_steps, obs_dim)) 
    eposide_a = torch.autograd.Variable(torch.randint(low=0, high=action_dim, size=(game_steps,))) 
    '''
    bayesian_update_instance = BayesianUpdater(opponent_model)

    adv_embedding = bayesian_update_instance.generate_cur_embedding()
    cur_belief = bayesian_update_instance.get_current_belief()
    print ('init adv_embedding', adv_embedding)
    print ('init cur_belief', cur_belief)

    bayesian_update_instance.update_belief(eposide_o,eposide_a)
    adv_embedding = bayesian_update_instance.generate_cur_embedding()
    cur_belief = bayesian_update_instance.get_current_belief()
    print ('adv_embedding after update ', adv_embedding)
    print ('cur_belief after update ', cur_belief)
    '''

    vi1 = VariationalInference(opponent_model, latent_dim=latent_dim, n_update_times=50, game_steps=game_steps)
    vi1.init_all()
    vi1.update(eposide_o,eposide_a)
    cur_ce_1 = vi1.get_cur_ce()
    cur_latent1_np = vi1.generate_cur_embedding(is_np=True)
    print ('ce loss', cur_ce_1)
    print ('latent emb', cur_latent1_np)


    # exp3 = EXP3(n_action=2, gamma=0.2, min_reward=-500, max_reward=100)
    # exp3.init_weight()
    # agent_selected = exp3.sample_action()
    # print ('agent_selected',agent_selected)
    # exp3.update(-100,agent_selected)
    # agent_selected = exp3.sample_action()
    # print ('agent_selected',agent_selected)
    # exp3.update(-60,agent_selected)
    # p = exp3.get_p()
    # print ('action select prob',p)




