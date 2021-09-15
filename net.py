import torch
import torch.nn as nn
import numpy as np

def cal_cos_similarity(x, y): # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
    cos_similarity = xy/x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity

def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size())

class IGMTF(nn.Module):
    def __init__(self, d_feat=6, hidden_size=256, num_layers=2, dropout=0, k_day = 10, n_neighbor=10):
        super().__init__()
        self.k_day =k_day  
        self.n_neighbor = n_neighbor

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.lins = nn.Sequential()
        self.lins.add_module("linear"+str(0), nn.Linear(hidden_size, 512))
        self.lins.add_module("leakyrelu"+str(0), nn.LeakyReLU())
        self.lins.add_module("linear"+str(1), nn.Linear(512, 512))
        self.lins.add_module("leakyrelu"+str(1), nn.LeakyReLU())
        self.lins.add_module("linear"+str(2), nn.Linear(512, hidden_size))
        self.lins.add_module("leakyrelu"+str(2), nn.LeakyReLU())

        self.fc_output = nn.Linear(hidden_size*2, hidden_size*2)
        self.project1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.project2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size*2, 1)

        self.leaky_relu = nn.LeakyReLU()
        self.d_feat = d_feat
     
    def forward(self, input, get_hidden=False, train_hidden=None, train_hidden_day = None):
        # x: [N, F*T]
        device = input.device

        x = input
        # x: [N, F*T]
        x = x.reshape(x.shape[0]*x.shape[1], -1)
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        mini_batch_out = self.lins(out)
        if get_hidden is True:
            return mini_batch_out

        mini_batch_out_day = torch.mean(mini_batch_out.reshape(input.shape[0], input.shape[1], -1), dim=1)
        day_similarity = cal_cos_similarity(mini_batch_out_day, train_hidden_day)
        day_index = torch.topk(day_similarity, self.k_day, dim=1)[1]
        sample_train_hidden = train_hidden[day_index,:,:]
        sample_train_hidden = sample_train_hidden.reshape(-1, sample_train_hidden.shape[-1]).to(device)
        cos_similarity = cal_cos_similarity(self.project1(mini_batch_out), self.project2(sample_train_hidden))
        
        
        row = torch.linspace(0,x.shape[0]-1, x.shape[0]).reshape([-1, 1]).repeat(1, self.n_neighbor).reshape(1, -1).to(device)
        column = torch.topk(cos_similarity, self.n_neighbor, dim = 1)[1].reshape(1, -1)
        mask = torch.sparse_coo_tensor(torch.cat([row, column]), torch.ones([row.shape[1]]).to(device)/self.n_neighbor, (x.shape[0], sample_train_hidden.shape[0]))
        cos_similarity = sparse_dense_mul(mask, cos_similarity)

        agg_out = torch.sparse.mm(cos_similarity, self.project2(sample_train_hidden))
        out = torch.cat([mini_batch_out, agg_out], axis=1)

        out = self.fc_out(out).squeeze()
        out = out.reshape([input.shape[0], -1])
        return out