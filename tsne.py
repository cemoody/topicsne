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


class TSNE(nn.Module):
    def __init__(self, n_points, n_topics, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(TSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_points, n_topics)

    def forward(self, pij, i, j):
        # Get  for all points
        x = self.logits.weight
        # Compute squared pairwise distances
        dkl2 = pairwise(x)
        # Compute partition function
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal
        # Compute the numerator
        xi = self.logits(i)
        xj = self.logits(j)
        num = ((1. + (xi - xj)**2.0).sum(1)).pow(-1.0).squeeze()
        # This probability is the probability of picking the (i, j)
        # relationship out of N^2 other possible pairs in the 2D embedding.
        qij = num / part.expand_as(num)
        # Compute KLD
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)
