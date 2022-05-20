import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss,NLLLoss,CrossEntropyLoss
from torch.distributions import Normal, Categorical, OneHotCategorical

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

# encoder for a auto encoder 
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        embedding = self.m_z(h)
        return embedding

# encoder for a VAE
class EncoderVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EncoderVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, 2*output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        h = F.relu(self.fc1(x))
        distributions = self.m_z(h)
        mu = distributions[:, :self.output_dim]
        logvar = distributions[:, self.output_dim:]
        embedding = reparametrize(mu, logvar)
        return embedding,mu,logvar

class Decoder(nn.Module):
    def __init__(self, input_dim1, hidden_dim, latent_dim, output_dim1):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim+latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim1)

    def forward(self, x, latent):
        if len(latent.size()) == 3:
            latent = latent.squeeze(0)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = torch.cat((h, latent), -1)
        h = F.relu(self.fc3(h))
        output = self.fc4(h)
        return output

class OpponentModel(nn.Module):
    def __init__(self, obs_dim, num_adv_pool, hidden_dim, latent_dim, 
                action_dim, encoder_weight_path, decoder_weight_path):
        super(OpponentModel, self).__init__()   

        self.num_adv_pool = num_adv_pool
        self.latent_dim = latent_dim
        self.encoder = EncoderVAE(num_adv_pool, hidden_dim, latent_dim)
        self.encoder.load_state_dict(torch.load(encoder_weight_path,map_location=torch.device('cpu')))
        self.encoder.eval()
        self.decoder = Decoder(obs_dim, hidden_dim, latent_dim, action_dim)
        self.decoder.load_state_dict(torch.load(decoder_weight_path,map_location=torch.device('cpu')))
        self.decoder.eval()
        self.cur_policy_vector = None
        self.use_policy_vec = False
        self.encoder_weight_path = encoder_weight_path
        self.decoder_weight_path = decoder_weight_path

    # inference using obs and opponent idx
    def inference_action(self, observation, adv_index):
        embedding,_ = self.sample_mean(adv_index,is_reduce_mean=False)
        output = self.decoder(observation, embedding)
        probs = F.softmax(output, dim=-1)
        pred_action = probs.data.max(1, keepdim=True)[1] 
        embedding = embedding.cpu().detach().numpy()
        embedding_out = np.mean(embedding, axis=0)
        return probs,pred_action,embedding_out

    # inference using obs and embedding
    def inference_action_by_emb(self, observation, embedding):
        output = self.decoder(observation, embedding)
        probs = F.softmax(output, dim=-1)
        pred_action = probs.data.max(1, keepdim=True)[1] 
        return probs,pred_action,output

    # sample embedding using opponent idx
    def sample_embedding(self, adv_index, is_reduce_mean):
        if not self.use_policy_vec:
            adv_index = F.one_hot(adv_index, num_classes=self.num_adv_pool).float()
        embedding,_,_ = self.encoder(adv_index)
        if is_reduce_mean:
            embedding = embedding.cpu().detach().numpy()
            embedding_out = np.mean(embedding, axis=0)
        else:
            embedding_out = embedding
        return embedding_out

    # sample embedding mean using opponent idx
    def sample_mean(self, adv_index,is_reduce_mean,is_use_policy_vec=False):
        if not is_use_policy_vec:
            adv_index = F.one_hot(adv_index, num_classes=self.num_adv_pool).float()
        embedding,mu,logvar = self.encoder(adv_index)
        if is_reduce_mean:
            mu = mu.cpu().detach().numpy()
            mu_out = np.mean(mu, axis=0)
            logvar = logvar.cpu().detach().numpy()
            logvar_out = np.mean(logvar, axis=0)
        else:
            mu_out = mu
            logvar_out = logvar
        return mu_out,logvar_out

    def set_policy_vector(self,policy_vector):
        self.cur_policy_vector = policy_vector
        self.use_policy_vec = True

    def save_parameters(self):
        torch.save(self.encoder.state_dict(), self.encoder_weight_path, _use_new_zipfile_serialization=False)
        torch.save(self.decoder.state_dict(), self.decoder_weight_path, _use_new_zipfile_serialization=False)




