# -*- coding: utf-8 -*-

import numpy as np
import numba

from .utils import check_pairwise_arrays

@numba.jit()
def weighted_lp_vec(x,y,weights=None,p=1):
    '''
    this is a "Dissimilarity Measure"
    parameters:
    x - a vector
    y - a vector

    refences:
    "The weighted lp metric DMs" from p604, chapter 11.2.2
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

  

def lp_mat(X,Y,p=1):

    '''
    this is a "Dissimilarity Measure"
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

def dG_vec(x,y,maxVec,minVec):
    '''
    this is a "Dissimilarity Measure"
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

def dQ_vec(x,y):
    '''
    this is a "Dissimilarity Measure"
    '''
    x,y = check_pairwise_arrays(x,y)
    l = x.shape[-1]
    diffVec = x-y
    sumVec = x+y
    tmp1 = np.power(diffVec/sumVec,2).sum()/l
    ans = np.sqrt(tmp1)

    return ans

def inner_vec(x,y):
    '''
    this is a "Similarity Measure"
    '''
    x,y = check_pairwise_arrays(x,y)
    ans = np.inner(x,y).squeeze()
    return ans

def cosine_vec(x,y):
    '''
    this is a "Similarity Measure"
    '''
    x,y = check_pairwise_arrays(x,y)
    numerator = np.inner(x,y).squeeze()
    denominator = np.sqrt(np.dot(x,x.T))*np.sqrt(np.dot(y,y.T))
    denominator = denominator.squeeze() 
    return numerator/denominator

def pearson_vec(x,y):
    '''
    this is a "Similarity Measure"
    '''
    x,y = check_pairwise_arrays(x,y)
    x, y = x-np.mean(x), y-np.mean(y)
    return cosine_vec(x,y)

def pearson_like_vec(x,y):
    '''
    this is a "Dissimilarity Measure"
    '''
    x,y = check_pairwise_arrays(x,y)
    ans = (1-pearson_SM_vec(x,y))/2
    return ans

def tanimoto_vec(x,y):
    '''
    this is a "Similarity Measure"
    '''
    '''
    x,y = check_pairwise_arrays(x,y)
    numerator = np.inner(x,y).squeeze()
    denominator = np.dot(x,x.T)+np.dot(y,y)-numerator
    denominator = denominator.squeeze()
    '''
    x,y = check_pairwise_arrays(x,y)
    numerator = 1
    diffVec = x-y
    denominator = 1 + np.dot(diffVec, diffVec.T)/np.dot(x, y.T)
    ans = numerator/denominator
    return ans

def unknow_vec(x,y):
    '''
    this is a "Similarity Measure"
    '''
    x,y = check_pairwise_arrays(x,y)
    xyLengthSum = np.dot(x,x.T)+np.dot(y,y.T)
    ans = 1 - weighted_lp_vec(x,y,weights=None,p=2)/xyLengthSum
    return ans

#=========belows are Discrete-Valued Vectors

def contingency_table_vec(x,y,k=None):

    '''
    performance:
    >>> timeit contingency_table_vec(np.arange(1000),np.arange(1000),1000)
    1 loop, best of 3: 1.61 s per loop
    >>> timeit contingency_table_vec(np.arange(2000),np.arange(2000),2000)
    1 loop, best of 3: 9.99 s per loop

    '''
    assert k != None, 'k should not be None'
    x,y = check_pairwise_arrays(x,y)   
    @numba.jit(nopython=True)
    def _inner(x,i,y,j):
        xBool = (x==i)
        yBool = (y==j)
        ans = (xBool &yBool).sum()
        return ans
        #return ((x==i)&(y==j)).sum() # the sentence maybe faster

    @numba.jit(nopython=True,parallel=True)
    def _contingency_table_vec(x,y,table,k=None):
        for i in range(k):
            for j in range(k):
                table[i,j]=_inner(x,i,y,j)
        return table    

    table = np.zeros([k,k])
    table = _contingency_table_vec(x,y,table,k)
    return table

def hamming_distance_vec(x,y):
    '''
    this is a "Dissimilarity Measure"
    '''
    x,y = check_pairwise_arrays(x,y)  
    k = np.max(np.max(x),np.max(y))
    table = contingency_table_vec(x,y,k)
    DH = table.sum() - (table*np.eye(k)).sum()
    return DH

def tanimoto_vec(x,y):
    '''
    this is a "Similarity Measure"
    '''
    x,y = check_pairwise_arrays(x,y)
    k = np.max(np.max(x),np.max(y))
    table = contingency_table_vec(x,y,k) 
    numerator = (table*np.eye(k)).sum()
    tableSum = table.sum()
    row0Sum = table[0,:].sum()
    col0Sum = table[:,0].sum()
    nx = tableSum - row0Sum
    ny = tableSum - col0Sum
    denominator = nx + ny - (tableSum-row0Sum-col0Sum+table[0,0])
    return numerator/denominator

# ==== Mixed Valued Vectors
def gowe_mix_vec(x,y,binVarInds=[],nomOrOrdVarInds=[],intOrRatVarInds=[],r=[]):
    '''
    parameters
    ----------
    b : should be a list, each element is the index of feature dimension that
    is a binary variable.

    example
    -------
    >>> x1 = np.array([1.2,1.5,1.9,0,1])
    >>> x2 = np.array([0.3,0.4,0.6,0,0])
    >>> intOrRatVarInds = [0,1,2]
    >>> r = [9.7,12.6,14.4]
    >>> binVarInds=[3]
    >>> nomOrOrdVarInds = [4]
    >>> gowe_mix_vec(x1,x2,binVarInds,nomOrOrdVarInds,intOrRatVarInds,r)
    0.6824
    '''
    #x,y = check_pairwise_arrays(x,y)
    x,y = x.squeeze(), y.squeeze()
    l = x.shape[0]
    assert len(binVarInds)+len(nomOrOrdVarInds)+len(intOrRatVarInds) == l,\
        '''the binVarInds,nomOrOrdVarInds,intOrRatVarInds length 
           not equal x feature dimension length'''
    assert len(intOrRatVarInds) == len(r),'the intOrRatVarInds length not equal r'
    #process the wieights 
    weights = np.ones_like(x)
    binVarInds = np.array(binVarInds)
    xNoneFlag = x == None
    yNoneFlag = y == None
    NoneDim = xNoneFlag | yNoneFlag
    weights[NoneDim] = 0
    weights[binVarInds] = 0
    denominator = weights.sum()

    if denominator == 0: return None
    #1 - process the binary variable dimension
    s = np.zeros_like(x)
    if binVarInds != []:  
        bDimTrue = (x[binVarInds]==1) & (y[binVarInds]==1)
        s[binVarInds] = bDimTrue
    #2 - process the nominal or ordinal variables
    if nomOrOrdVarInds != []:
        noDimTrue = x == y
        s[nomOrOrdVarInds] = noDimTrue[nomOrOrdVarInds]
    #3 - process the interval or ratio scaled variables
    if intOrRatVarInds != []:
        r = np.asarray(r)
        s[intOrRatVarInds] = 1-np.abs(x[intOrRatVarInds]-y[intOrRatVarInds])/r
    numerator = s.sum()
    ans = numerator/denominator
    return ans

if __name__ == '__main__':
    pass