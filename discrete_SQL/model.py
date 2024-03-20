import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class QTransformer(nn.Module):
    def __init__(self, s_dim, a_dim, a_bins, alpha, num_layers=4, nhead=2, action_min=-1.0, action_max=1.0, hdim=1024):
        super(QTransformer, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bins = a_bins
        self.alpha = alpha
        self.num_layers = num_layers
        self.nhead = nhead
        self.action_min = action_min
        self.action_max = action_max
        
        self.s_embed = nn.Linear(s_dim, hdim)
        self.a_embed = nn.Linear(1, hdim)
        #self.a_embed = nn.Embedding(a_bins, hdim)
        self.positional_encoding = nn.Embedding(a_dim, hdim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead, dim_feedforward=hdim, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.q_head = nn.Linear(hdim, a_bins)
        
    def quantize_action(self, a):
        rescaled_action = (a - self.action_min) / (self.action_max - self.action_min) * (self.a_bins - 1)
        discrete_action = torch.round(rescaled_action).long()
        return discrete_action
    
    def dequantize_action(self, a):
        #a = torch.argmax(a, dim=2, keepdim=True)
        bin_width = (self.action_max - self.action_min) / self.a_bins
        continuous_action = self.action_min + (a + 0.5) * bin_width
        return continuous_action
        
    def forward(self, s, a=None, is_causal=True):
        # s: [batch_size, 1, s_dim]
        # a: [batch_size, t, 1]
        x = self.s_embed(s)
        seq_len = 1
        if a is not None:
            #a = self.quantize_action(a)
            a = self.a_embed(a) + self.positional_encoding.weight[:a.shape[1], :].unsqueeze(0)
            #a = self.a_embed(a.squeeze(-1)) + self.positional_encoding.weight[:a.shape[1], :].unsqueeze(0)
            x = torch.cat([x, a], dim=1)
            seq_len = a.shape[1] + 1
        x = self.transformer(x, mask=torch.nn.Transformer.generate_square_subsequent_mask(seq_len).cuda())#, is_causal=is_causal)
        Q = self.q_head(x)
        V = torch.logsumexp(Q / self.alpha, dim=2) * self.alpha
        pi = F.softmax(Q / self.alpha, dim=2)
        return Q, V, pi
    
    def sample_action(self, s, return_entropy=False, exploration_alpha=0.1):
        alpha_temp = self.alpha
        self.alpha = exploration_alpha
        s = s.unsqueeze(1)
        a = None
        for i in range(self.a_dim):
            _, _, pi = self.forward(s, a)
            dist = torch.distributions.Categorical(pi)
            a_i = dist.sample().unsqueeze(2)
            a_i = self.dequantize_action(a_i)
            if a is None:
                a = a_i
            else:
                a = torch.cat([a, a_i[:, -1:]], dim=1)
        self.alpha = alpha_temp
        if return_entropy:
            entropy = -dist.log_prob(self.quantize_action(a_i).squeeze(2))
            return a.squeeze(2), entropy.detach().sum(1).mean()
        return a.squeeze(2)
    
    
class QMLP(nn.Module):
    def __init__(self, state_dim, h_dim, a_bins):
        super(QMLP, self).__init__()
        self.state_dim = state_dim
        
        self.fc1 = nn.Linear(state_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, a_bins)

        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        
    def forward(self, s):
        x = F.relu(self.ln1(self.fc1(s)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
# AutoRegressive Q function
class ARQ(nn.Module):
    def __init__(self, s_dim, a_dim, a_bins, alpha, action_min=-1.0, action_max=1.0):
        super(ARQ, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bins = a_bins
        self.alpha = alpha
        self.action_min = action_min
        self.action_max = action_max
        
        self.mlp = nn.ModuleList([QMLP(s_dim+i,512,a_bins) for i in range(a_dim)])
        self.q_final = QMLP(s_dim+a_dim, 512, 1)
        
    def quantize_action(self, a):
        rescaled_action = (a - self.action_min) / (self.action_max - self.action_min) * (self.a_bins - 1)
        discrete_action = torch.round(rescaled_action).long()
        return discrete_action
    
    def dequantize_action(self, a):
        #a = torch.argmax(a, dim=2, keepdim=True)
        bin_width = (self.action_max - self.action_min) / self.a_bins
        continuous_action = self.action_min + (a + 0.5) * bin_width
        return continuous_action
        
    def forward_once(self, s, a=None):
        mlp_idx = 0
        if a is not None:
            s = torch.cat([s, a], dim=1)
            mlp_idx = a.shape[1]
        Q = self.mlp[mlp_idx](s)
        V = torch.logsumexp(Q / self.alpha, dim=1, keepdim=True) * self.alpha
        pi = F.softmax(Q / self.alpha, dim=1)
        return Q, V, pi
    
    def forward(self, s, a):
        Q = self.mlp[0](s)
        Q = Q.unsqueeze(1)
        for i in range(1, self.a_dim):
            Q_i = self.mlp[i](torch.cat([s, a[:, :i]], dim=1))
            Q  = torch.cat([Q, Q_i.unsqueeze(1)], dim=1)
        Q_final = None#self.q_final(torch.cat([s, a], dim=1))
        V = torch.logsumexp(Q / self.alpha, dim=2) * self.alpha
        pi = F.softmax(Q / self.alpha, dim=2)
        return Q, V, pi, Q_final
    
    def sample_action(self, s, return_entropy=False, exploration_alpha=0.1):
        alpha_temp = self.alpha
        self.alpha = exploration_alpha
        a = None
        entropy = 0.0
        for i in range(self.a_dim):
            _, _, pi = self.forward_once(s, a)
            dist = torch.distributions.Categorical(pi)
            a_i = dist.sample().unsqueeze(1)
            if return_entropy:
                entropy = entropy + -dist.log_prob(a_i.squeeze(1))
            a_i = self.dequantize_action(a_i)
            if a is None:
                a = a_i
            else:
                a = torch.cat([a, a_i], dim=1)
        self.alpha = alpha_temp
        if return_entropy:
            return a, entropy.detach()#.mean()
        return a