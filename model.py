import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn


def gumbel_sample(logits, tau=1.0, temperature=0.0):
    # Uniform sample
    noise = torch.rand(logits.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    gumbel = Variable(noise)
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
    def __init__(self, pij, n_obs, n_topics, n_dim):
        self.n_obs = n_obs
        self.n_dim = n_dim
        super(TopicSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_obs, n_topics)
        # Vector for each topic
        self.topic = nn.Linear(n_topics, n_dim)

    def dirichlet_ll(self):
        pass

    def forward(self, pij, i, j):
        # Get  for all points
        x = self.topic(gumbel_sample(self.logits.weight))
        # Compute squared pairwise distances
        dkl2 = pairwise(x)
        # Compute partition function
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal
        # Compute the numerator
        xi = self.topic(gumbel_sample(self.logits(i)))
        xj = self.topic(gumbel_sample(self.logits(j)))
        num = ((1. + (xi - xj)**2.0).sum(1)).pow(-1.0)
        qij = num / part
        # Compute KLD
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        # Compute Dirichlet likelihood
        # loss_dir = self.dirichlet_ll()
        return loss_kld # + loss_dir

    def __call__(self, *args):
        return self.forward(*args)
