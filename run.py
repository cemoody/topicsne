from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform

from joblib import Memory

mem = Memory(cachedir='tmp')


@mem.cache
def preprocess(perplexity=30, metric='euclidean'):
    """ Compute pairiwse probabilities for MNIST pixels.
    """
    digits = datasets.load_digits(n_class=6)
    pos = digits.data
    distances2 = pairwise_distances(pos, metric=metric, squared=True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    # Convert to n x n prob array
    pij = squareform(pij)
    return pij



