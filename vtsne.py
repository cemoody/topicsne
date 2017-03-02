import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

import numpy as np


def pairwise(data):
    n_obs, dim = data.size()
    xk = data.unsqueeze(0).expand(n_obs, n_obs, dim)
    xl = data.unsqueeze(1).expand(n_obs, n_obs, dim)
    dkl2 = ((xk - xl)**2.0).sum(2).squeeze()
    return dkl2


class VTSNE(nn.Module):
    def __init__(self, n_points, n_topics, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(VTSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits_mu = nn.Embedding(n_points, n_topics)
        self.logits_lv = nn.Embedding(n_points, n_topics)
    
    @property
    def logits(self):
        return self.logits_mu

    def reparametrize(self, mu, logvar):
        # From VAE example
        # https://github.com/pytorch/examples/blob/master/vae/main.py
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld = torch.sum(kld).mul_(-0.5)
        return z, kld

    def sample_logits(self, i=None):
        if i is None:
            return self.reparametrize(self.logits_mu.weight, self.logits_lv.weight)
        else:
            return self.reparametrize(self.logits_mu(i), self.logits_lv(i))

    def forward(self, pij, i, j):
        # Get  for all points
        x, loss_kldrp = self.sample_logits()
        # Compute squared pairwise distances
        dkl2 = pairwise(x)
        # Compute partition function
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal
        # Compute the numerator
        xi, _ = self.sample_logits(i)
        xj, _ = self.sample_logits(j)
        num = ((1. + (xi - xj)**2.0).sum(1)).pow(-1.0).squeeze()
        qij = num / part.expand_as(num)
        # Compute KLD(pij || qij)
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        # Compute sum of all variational terms
        return loss_kld.sum() + loss_kldrp.sum() * 1e-7

    def __call__(self, *args):
        return self.forward(*args)
