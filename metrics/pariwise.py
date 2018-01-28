# -*- coding: utf-8 -*-

import numpy as np
from functools import partial
from scipy.sparse import issparse
import numba 
#import pp #parallel python

from .utils import check_pairwise_arrays

def euclidean_distances(X,Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::
        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))


    performance:
    >>> X = np.random.randn(1000,1000)

    >>> from scipy.spatial.distance as cdist
    >>> timeit cdist(X,X,'euclidean')
    1 loop, best of 3: 1.08s per loop

    >>> timeit euclidean_distances(X,X)
    10 loops, best of 3: 182 ms per loop

    >>> from sklearn.metrics.pairwise import euclidean_distances as ed
    >>> timeit ed(X,X)
    10 loops, best of 3: 161 ms per loop
    """ 
    X,Y = check_pairwise_arrays(X,Y)

    X_square = X*X
    XX = X_square.sum(1).reshape(-1,1)

    Y_square = Y*Y
    YY = Y_square.sum(1).reshape(1,-1)

    XY_dot = np.dot(X,Y.T)

    distances = XX -2*XY_dot+YY

    np.maximum(distances, 0, out=distances)

    distances = np.sqrt(distances)

    return distances


def manhattan_distances():
    pass
def cosine_distances(X, Y=None):
    pass
if __name__ == '__main__':

    X = np.random.randn(2,2)
    distances = euclidean_distances(X,X)
    print(distances)

    
