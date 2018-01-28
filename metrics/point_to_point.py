# -*- coding: utf-8 -*-

import numpy as np
import numba

from .utils import check_pairwise_arrays

@numba.jit()
def weighted_lp_DM_vec(x,y,weights=None,p=1):

    assert p>0, 'p must bigger than 0'
    x,y = check_pairwise_arrays(x,y)

    if not weights: 
        weights = np.ones_like(x)
    else:
        x,weights = check_pairwise_arrays(x,weights)

    absVec = np.power(x-y,p)
    wAbsVec = weights*absVec
    sumWAbsVec = wAbsVec.sum()
    distanceP = np.power(sumWAbsVec,1.0/p)
    return distanceP


def weighted_lp_DM_mat(X,Y):
    pass

if __name__ == '__main__':
    pass