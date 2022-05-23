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


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        embedding = self.m_z(h)
        return embedding


# multi var version
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
    def __init__(self, input_dim1, hidden_dim, latent_dim, output_dim1, output_dim2=None):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim+latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim1)

        self.is_disc_head = False
        if output_dim2 is not None:
            self.is_disc_head = True
            self.fc5 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, x, latent):
        if len(latent.size()) == 3:
            latent = latent.squeeze(0)
        # x.requires_grad = True
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = torch.cat((h, latent), -1)
        h = F.relu(self.fc3(h))
        # probs = F.softmax(self.fc5(h), dim=-1)
        output = self.fc4(h)

        if self.is_disc_head:
            output2 = self.fc5(h)
            return output, output2
        # return output, None
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


    def inference_action(self, observation, adv_index):
        # embedding = self.sample_embedding(adv_index,is_reduce_mean=False)
        embedding = self.sample_mean(adv_index,is_reduce_mean=False)

        output = self.decoder(observation, embedding)
        probs = F.softmax(output, dim=-1)
        pred_action = probs.data.max(1, keepdim=True)[1] 
        
        embedding = embedding.cpu().detach().numpy()
        embedding_out = np.mean(embedding, axis=0)
        return probs,pred_action,embedding_out

    def inference_action_by_emb(self, observation, embedding):
        output = self.decoder(observation, embedding)
        probs = F.softmax(output, dim=-1)
        pred_action = probs.data.max(1, keepdim=True)[1] 

        return probs,pred_action,output

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

    def sample_mean(self, adv_index, is_reduce_mean):
        if not self.use_policy_vec:
            adv_index = F.one_hot(adv_index, num_classes=self.num_adv_pool).float()
        embedding,mu,_ = self.encoder(adv_index)
        if is_reduce_mean:
            mu = mu.cpu().detach().numpy()
            mu_out = np.mean(mu, axis=0)
        else:
            mu_out = mu
        return mu_out

    def action(self, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        dim_c = 2

        policy_vector = self.cur_policy_vector
        # embedding = self.sample_mean(adv_index,is_reduce_mean=False)
        embedding = self.sample_embedding(policy_vector,is_reduce_mean=False)

        output = self.decoder(observation, embedding)
        probs = F.softmax(output, dim=-1)
        pred_action = probs.data.max(1, keepdim=True)[1] 
        
        embedding = embedding.cpu().detach().numpy()
        embedding_out = np.mean(embedding, axis=0)

        c = Categorical(probs)
        action = c.sample()
        u = np.zeros(5)
        u[action.item()] += 1
        return np.concatenate([u, np.zeros(dim_c)])

    def set_policy_vector(self,policy_vector):
        self.cur_policy_vector = policy_vector
        self.use_policy_vec = True

if __name__ == '__main__':

    obs_dim = 16 # observation dimension
    num_adv_pool = 4 # policies pool size
    action_dim = 7 
    hidden_dim = 128
    latent_dim = 2
    batch_size = 8
    encoder_weight_path = 'saved_model_params/encoder_vae_param.pt'
    decoder_weight_path = 'saved_model_params/decoder_param.pt'

    encoder = EncoderVAE(num_adv_pool, hidden_dim, latent_dim)
    encoder.load_state_dict(torch.load(encoder_weight_path,map_location=torch.device('cpu')))
    encoder.eval()

    decoder = Decoder(obs_dim, hidden_dim, latent_dim, action_dim, output_dim2=None)
    decoder.load_state_dict(torch.load(decoder_weight_path,map_location=torch.device('cpu')))
    decoder.eval()


    adv_index = torch.tensor([3]*batch_size) 
    onehot_index = F.one_hot(adv_index, num_classes=num_adv_pool).float()
    observation = torch.autograd.Variable(torch.rand(batch_size, obs_dim))  

    embedding,_,_ = encoder(onehot_index)
    print ('embedding', embedding.size())

    probs = decoder(observation, embedding)
    print ('decoder output',probs.size())

    opponent_model = OpponentModel(obs_dim, num_adv_pool, hidden_dim, latent_dim, 
                action_dim, encoder_weight_path, decoder_weight_path)

    raw_output,pred_action, _ = opponent_model.inference_action(observation, adv_index)
    print ('pred_action',pred_action.size(),pred_action)




