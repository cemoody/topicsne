from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform

from joblib import Memory
from wrapper import Wrapper
from tsne import TSNE

import numpy as np

mem = Memory(cachedir='tmp', verbose=100)


@mem.cache
def preprocess(perplexity=30, metric='euclidean'):
    """ Compute pairiwse probabilities for MNIST pixels.
    """
    digits = datasets.load_digits(n_class=6)
    pos = digits.data
    y = digits.target
    n_points = pos.shape[0]
    distances2 = pairwise_distances(pos, metric=metric, squared=True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    # Convert to n x n prob array
    pij = squareform(pij)
    return n_points, pij, y


n_points, pij2d, y = preprocess()
i, j = np.indices(pij2d.shape)
i = i.ravel()
j = j.ravel()
pij = pij2d.ravel().astype('float32')
# Remove self-indices
idx = i != j
i, j, pij = i[idx], j[idx], pij[idx]

n_topics = 2
n_dim = 2
print(n_points, n_dim, n_topics)

model = TSNE(n_points, n_topics, n_dim)
w = Wrapper(model, batchsize=4096, epochs=1)
for itr in range(500):
    w.fit(pij, i, j)

    # Visualize the results
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pos = model.logits.weight.cpu().data.numpy()
    f = plt.figure()
    plt.scatter(pos[:, 0], pos[:, 1], c=y * 1.0 / y.max())
    plt.axis('off')
    plt.savefig('scatter_{:03d}.png'.format(itr), bbox_inches='tight')
    plt.close(f)
