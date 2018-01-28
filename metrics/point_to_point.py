# -*- coding: utf-8 -*-

import numpy as np
import numba

from .utils import check_pairwise_arrays

@numba.jit()
def weighted_lp_DM_vec(x,y,weights=None,p=1):
    '''
    parameters:
    x - a vector
    y - a vector
    '''

    assert p>0, 'p must bigger than 0'
    x,y = check_pairwise_arrays(x,y)

    if not weights: 
        weights = np.ones_like(x)
    else:
        x,weights = check_pairwise_arrays(x,weights)

    absVec = np.power(np.abs(x-y),p)
    wAbsVec = weights*absVec
    sumWAbsVec = wAbsVec.sum()
    distanceWP = np.power(sumWAbsVec,1.0/p)
    return distanceWP


def lp_DM_mat(X,Y,p=1):

    '''
    lp_DM_mat(X,Y,1) is equal :
    >>> from sklearn.metrics.pairwise import manhattan_distances
    >>> manhattan_distances(X,Y,sum_over_features=True))

    outputs:
    distanceP: n_sample_X by n_sample_Y matrix

    example:
    >>> X = np.ones([1,2])
    >>> Y = 2*np.ones([2,2])
    >>> lp_DM_mat(X,Y)
    [[2,2]]
    '''
    assert p>0, 'p must bigger than 0'
    X,Y = check_pairwise_arrays(X,Y)
    
    D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    absMat = np.power(np.abs(D),p)
    sumAbsMat = absMat.sum(2)
    distanceP = np.power(sumAbsMat,1.0/p)

    return distanceP

def dG_DM_vec(x,y,maxVec,minVec):
    '''
    parameters:
    x - a vector
    y - a vector
    maxVec - a vector
    minVec - a vector
    '''
    x,y = check_pairwise_arrays(x,y)
    maxVec,minVec = check_pairwise_arrays(maxVec,minVec)
    l = x.shape[-1]
    absVec = np.abs(x-y)
    diffVec = maxVec - minVec
    tmp1 = 1 - np.sum(absVec/diffVec)/l
    ans = -np.log10(tmp1)

    return ans

def dQ_DM_vec(x,y):
    x,y = check_pairwise_arrays(x,y)
    l = x.shape[-1]
    diffVec = x-y
    sumVec = x+y
    tmp1 = np.power(diffVec/sumVec,2).sum()/l
    ans = np.sqrt(tmp1)

    return ans


if __name__ == '__main__':
    pass