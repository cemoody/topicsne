# Build a topic-like embedding for t-SNE
#     0. Measure distances d_ij of points
# 
#     1. Construct the input p_ij
# 
#     2. Define qij = (1 + dij2)^-1  /  SUM(over k, l, k!=l) (1 + dkl2)^-1
#               qij = (1 + dij2)^-1  /  (-N + SUM(over k, l) (1 + dkl2)^-1)
#               qij = (1 + dij2)^-1  /  Z
#           dij2 = ||x_i - x_j||^2
#           Where x_i = gs(r_i) . M
#           r_i = is a loading of a document onto topics
#           M = translation from topics to vector space
#           gs = gumbel-softmax of input rep
# 
#     3. Algorithm:
#       3.a Precompute p_ij
#       3.b Build pairwise matrix Sum dkl2
#           For all points, sample x_i = gs(r_i) . M
#           Build N^2 matrix of pairwise distances:  dkl2 = ||xk||^2 + ||xl||^2 - 2 xk . xl
#           Z = Sum over all, then subtract N to compensate for diagonal entries
#       3.c For input minibatch of ij, minimize p_ij (log(p_ij) - log(q_ij))
#     3. SGD minimize p_ij log(p_ij / q_ij)

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

import numpy as np


def gumbel_sample(logits, tau=1.0, temperature=0.0):
    # Uniform sample
    with torch.cuda.device(logits.get_device()):
        noise = torch.rand(logits.size())
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        gumbel = Variable(noise).cuda()
    sample = (logits + gumbel) / tau + temperature
    sample = F.softmax(sample.view(sample.size(0), -1))
    return sample.view_as(logits)


def pairwise(data):
    n_obs, dim = data.size()
    xk = data.unsqueeze(0).expand(n_obs, n_obs, dim)
    xl = data.unsqueeze(1).expand(n_obs, n_obs, dim)
    dkl2 = ((xk - xl)**2.0).sum(2).squeeze()
    return dkl2


class TopicSNE(nn.Module):
    def __init__(self, n_points, n_topics, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(TopicSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_points, n_topics)
        # Vector for each topic
        self.topic = nn.Linear(n_topics, n_dim)

    def positions(self):
        # x = self.topic(F.softmax(self.logits.weight))
        x = self.logits.weight
        return x

    def dirichlet_ll(self):
        pass

    def forward(self, pij, i, j):
        # Get  for all points
        with torch.cuda.device(pij.get_device()):
            alli = torch.from_numpy(np.arange(self.n_points))
            alli = Variable(alli).cuda()
        # x = self.topic(gumbel_sample(self.logits(alli)))
        x = self.logits(alli)
        # Compute squared pairwise distances
        dkl2 = pairwise(x)
        # Compute partition function
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal
        # Compute the numerator
        # xi = self.topic(gumbel_sample(self.logits(i)))
        # xj = self.topic(gumbel_sample(self.logits(j)))
        xi = self.logits(i)
        xj = self.logits(j)
        num = ((1. + (xi - xj)**2.0).sum(1)).pow(-1.0).squeeze()
        qij = num / part.expand_as(num)
        # Compute KLD
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        # Compute Dirichlet likelihood
        # loss_dir = self.dirichlet_ll()
        return loss_kld.sum() # + loss_dir

    def __call__(self, *args):
        return self.forward(*args)
