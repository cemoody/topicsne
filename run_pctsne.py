from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from torch import from_numpy
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import matplotlib.pyplot as plt

from pctsne import PCTSNE

digits = datasets.load_digits(n_class=6)
pos = digits.data
y = digits.target

# Find k nearest neighbors of pos
nbrs = NearestNeighbors(n_neighbors=16, algorithm="ball_tree").fit(pos)
distances, indices = nbrs.kneighbors(pos)
distances = distances[:, 1:]
indices = indices[:, 1:]
qry_vecs = from_numpy(pos).float()
knn_vecs = from_numpy(pos[indices]).float()


def dataloader(*arrs, batch_size=2048):
    dataset = TensorDataset(*arrs)
    arr_size = len(arrs[0])
    bs = BatchSampler(
        SequentialSampler(range(arr_size)), batch_size=batch_size, drop_last=False
    )
    return DataLoader(dataset, batch_sampler=bs, shuffle=True)


train = dataloader(from_numpy(qry_vecs), from_numpy(knn_vecs))

model = PCTSNE()
logger = WandbLogger(name="run01", project="pctsne")
trainer = pl.Trainer(
    limit_train_batches=1, max_epochs=1, progress_bar_refresh_rate=1, logger=logger
)

for itr in range(500):
    # Visualize the results
    trainer.fit(model=model, train_dataloaders=train)
    embed = model.trasform(qry_vecs)
    f = plt.figure()
    plt.scatter(embed[:, 0], embed[:, 1], c=y * 1.0 / y.max())
    plt.axis("off")
    plt.savefig(f"scatter_{itr:03d}.png", bbox_inches="tight")
    plt.close(f)
