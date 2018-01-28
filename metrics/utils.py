# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import issparse


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

def check_pairwise_arrays(X,Y):

    X, Y, dtype_float = _return_float_dtype(X, Y)
    x_shape = X.shape
    y_shape = Y.shape   
    assert x_shape == y_shape, 'X shape not equal Y shape'
    if X.ndim == 1: X = X.reshape(1,-1)
    if Y.ndim == 1: Y = Y.reshape(1,-1)
    
    return X,Y 