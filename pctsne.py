"""
The idea here is to learn a parametric mapping from
embeddings to a 2D space. 
"""
import torch
from torch import nn
from torch import optim, nn
import pytorch_lightning as pl


def log_cauchy(x):
    return -torch.log((1.0 + x**2.0))


def cauchy(x):
    return 1.0 / (1.0 + x**2.0)


class PCTSNE(pl.LightningModule):
    def __init__(
        self,
        n_dim=768,
        n_out=2,
        n_hi=32,
        c_ord=1.0,
        c_rpl=1.0,
        c_ls=1.0,
        c_l2=1.0,
        p=0.01,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_out = n_out
        self.n_hi = n_hi
        self.lin_2d = nn.Linear(n_dim, n_out)
        self.lin_nd = nn.Linear(n_dim, n_hi)
        self.lin_fin = self.lin_2d
        self.n_fin = n_out
        self.transform_ = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.Mish(),
            nn.Dropout(p),
            nn.Linear(n_dim, n_dim),
            nn.Mish(),
            nn.Dropout(p),
            nn.Linear(n_dim, n_dim),
            nn.Mish(),
        )
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.c_ord = c_ord
        self.c_rpl = c_rpl
        self.c_ls = c_ls
        self.c_l2 = c_l2

    def transform(self, x):
        return self.lin_fin(self.transform_(x))

    def transform_broadcast(self, qry_vec, knn_vec):
        # ivec is size (bs, n_dim)
        # kvec is size (bs, k, n_dim)
        k = knn_vec.size()[1]
        bs = qry_vec.size()[0]
        # flatten so we can transform knn vec
        knn_vec_ = knn_vec.reshape(bs * k, self.n_dim)
        xi_ = self.transform(qry_vec)
        xj_ = self.transform(knn_vec_)
        xi = xi_.unsqueeze(1).expand(bs, k, self.n_fin)
        xj = xj_.reshape(bs, k, self.n_fin)
        return xi, xj

    def compute_distances(self, xi, xj):
        # Compute the distance between each query and
        # each of its neighbors
        dik = torch.norm(xi - xj, dim=2)  # (bs, k)

        # Compute distance between the nth neighbors
        # in one item against the nth neighbor in random item in batch
        xj1 = xj.unsqueeze(1)  # (bs, 1, k, n_dim)
        xj2 = xj.unsqueeze(0)  # (1, bs, k, n_dim)
        dij = torch.norm(xj1 - xj2, dim=3)  # (bs, bs, k)
        return dik, dij

    def loss_ordinal(self, dik):
        """Enforce that neighbor k is closer than neighbor k+1"""
        # dij is size (bs, k)
        # diff is negative if first neighbor is closer than second
        # e.g. we want diff to be as negative as possible
        diff = torch.ravel(dik[:, 1:] - dik[:, :-1])  # (bs, k) -> (bs * k)
        # this will push diffs to be as negative as possible
        # because targets are zero
        loss = self.bce(diff, torch.zeros_like(diff))
        return loss

    def loss_repel(self, dik, dij):
        """Enforce that query is closer to neighbor than
        query is close to a random picked item"""
        # dik is distance from query to neighbor (bs, k)
        # dij is distance from neighbor to random item in batch (bs, bs, k)
        dik_ = dik.unsqueeze(1)  # (bs, 1, k)
        # When diff is negative, dik is closer than dij
        # we want diff to be as negative as possible
        diff = torch.ravel(dik_ - dij)  # (bs, bs, k) -> (bs * bs * k)
        loss = self.bce(diff, torch.zeros_like(diff))
        return loss

    def loss_l2(self, xi):
        """L2 loss on the embedding"""
        loss = torch.norm(xi, dim=2).mean()
        return loss

    def forward(self, qry_vec, knn_vec):
        bs, k, n_dim = knn_vec.size()
        # qry_vec (bs, n_dim)    -> xi (bs, k, n_dim)
        # knn_vec (bs, k, n_dim) -> xj (bs, k, n_dim)
        xi, xj = self.transform_broadcast(qry_vec, knn_vec)
        # dik is distance from query to its neighbors
        # dij is distance from neighbor to random item in batch
        dik, dij = self.compute_distances(xi, xj)

        loss_ord = self.loss_ordinal(dik) / (bs * k)
        loss_rpl = self.loss_repel(dik, dij) / (bs * bs * k)
        loss_xi = self.loss_l2(xi) / (bs * k)
        loss_xj = self.loss_l2(xj) / (bs * k)

        self.log("loss_ord", loss_ord)
        self.log("loss_rpl", loss_rpl)
        self.log("loss_xi", loss_xi)
        self.log("loss_xj", loss_xj)

        loss = (
            self.c_ord * loss_ord
            + self.c_rpl * loss_rpl
            + self.c_ls * loss_xi
            + self.c_l2 * loss_xj
        )
        self.log("loss", loss)
        return loss

    def __call__(self, *args):
        return self.forward(*args)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        qry_vec, knn_vec = batch
        loss = self.forward(qry_vec, knn_vec)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
