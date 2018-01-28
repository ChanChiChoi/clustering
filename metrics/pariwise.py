# -*- coding: utf-8 -*-

import numpy as np
from functools import partial
from scipy.sparse import issparse
import numba 

# Utility Functions
# the function borrow from 
#https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/pairwise.py
def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float

    return X, Y, dtype

def _check_pairwise_arrays(X,Y):

    X, Y, dtype_float = _return_float_dtype(X, Y)
    x_shape = X.shape
    y_shape = Y.shape   
    assert x_shape == y_shape, 'X shape not equal Y shape'
    if X.ndim == 1: X = X.reshape(1,-1)
    if Y.ndim == 1: Y = Y.reshape(1,-1)
    return X,Y 

def euclidean_distances(X,Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::
        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
    """ 
    X,Y = _check_pairwise_arrays(X,Y)

    X_square = X*X
    XX = X_square.sum(1).reshape(-1,1)

    Y_square = Y*Y
    YY = Y_square.sum(1).reshape(1,-1)

    XY_dot = np.dot(X,Y.T)

    distances = XX -2*XY_dot+YY

    np.maximum(distances, 0, out=distances)

    distances = np.sqrt(distances)

    return distances

if __name__ == '__main__':

