from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from torch import from_numpy
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

from pctsne import PCTSNE

batch_size = 128
c_rpl = 10
c_l2 = 1.0
do_rpl_cycle = False
do_lin_fin_cycle = True

digits = datasets.load_digits(n_class=10)
pos = digits.data
y = digits.target

# Find k nearest neighbors of pos
nbrs = NearestNeighbors(n_neighbors=16, algorithm="ball_tree").fit(pos)
distances, indices = nbrs.kneighbors(pos)
distances = distances[:, 1:]
indices = indices[:, 1:]
qry_vecs = from_numpy(pos).float()
knn_vecs = from_numpy(pos[indices]).float()


def dataloader(*arrs, batch_size=1024):
    dataset = TensorDataset(*arrs)
    arr_size = len(arrs[0])
    bs = BatchSampler(
        RandomSampler(range(arr_size)), batch_size=batch_size, drop_last=False
    )
    return DataLoader(dataset, batch_sampler=bs, shuffle=False)


train = dataloader(qry_vecs, knn_vecs, batch_size=batch_size)


model = PCTSNE(n_dim=64, c_rpl=c_rpl)
logger = WandbLogger(name="run01", project="pctsne")


class ScatterPlot(Callback):
    last = None

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        embed = model.transform(qry_vecs).detach().numpy()
        f = plt.figure()
        plt.scatter(embed[:, 0], embed[:, 1], c=y * 1.0 / y.max())
        plt.axis("off")
        plt.savefig(f"scatter_{epoch:03d}.png", bbox_inches="tight")
        plt.close(f)
        if do_rpl_cycle:
            if int(epoch / 100) % 2 == 0:
                # on even centuries turn on repulsive loss
                model.c_rpl = c_rpl
                model.c_l2 = c_l2
            else:
                # on odd centuries turn off repulsive loss
                # and turn off central pressure from l2
                model.c_rpl = c_rpl * 0
                model.c_l2 = c_l2 * 0
        if do_lin_fin_cycle:
            if int(epoch / 100) % 2 == 0:
                # on even centuries use 2D
                model.lin_fin = model.lin_2d
                model.n_fin = model.n_out
                if self.last != model.n_out:
                    print("Switching to 2D")
                    self.last = model.n_out
                    model.lin_fin.reset_parameters()
            else:
                # on odd centuries use ND
                model.lin_fin = model.lin_nd
                model.n_fin = model.n_hi
                if self.last != model.n_hi:
                    print("Switching to ND")
                    self.last = model.n_hi


trainer = pl.Trainer(max_epochs=700, logger=logger, callbacks=[ScatterPlot()])
trainer.fit(model=model, train_dataloaders=train)
