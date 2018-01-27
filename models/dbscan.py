# -*- coding:utf-8 -*-
'''
the most original code borrowed from 
https://github.com/choffstein/dbscan/blob/master/dbscan/dbscan.py
I will improve the whole code for speed and efficiency

'''
import time
import numpy as np
from functools import partial
import numba

__VERSION = '0.1.0'
#@numba.jit()
def _euclidean(p,q):
    return np.sqrt(np.power(p-q,2).sum())

#@numba.jit()
def _isNeighborhood(q, p, eps, metricFn):
    return metricFn(p,q) < eps    

#TODO:the function should use KNN algorithm
def _region_query(curSampleInd, XMat, eps, metricFn):
    isNeighborhood = partial(_isNeighborhood, p=XMat[curSampleInd,:], eps=eps, metricFn=metricFn)
    nSamples = XMat.shape[0]
    
    seeds = list(filter(lambda ind:isNeighborhood(XMat[ind,:]), range(nSamples)))
    return seeds


def _curSample_expand(XMat, labels, metricFn,
                    curSampleInd, curLabel,
                    eps, minSamples):
    ''' 
    if one point is classifitied by one cluster, and the point is a noise point,
    although the point will be classifitied twice by two different cluster, 
    but others point connetced by the noise point should not be effect, because they
    could not be propagated.
    '''    
    region_query = partial(_region_query, XMat=XMat, eps=eps, metricFn=metricFn)
    
    #get the current sample's neighborhood region
    seeds = region_query(curSampleInd)
    
    if len(seeds) < minSamples:
        labels[curSampleInd] = -1#NOISE point
        return False
    else:
        labels[curSampleInd] = curLabel
        for seed in seeds:
            labels[seed] = curLabel

        #TODO: the queue should be handled parallel
        #1 - maintain one queue of current sample's propagating region 
        while seeds:
            curSeed = seeds[0]
            curSeedNeigh = region_query(curSeed)
            
            if len(curSeedNeigh) > minSamples:
                for seedNeighInd in range(len(curSeedNeigh)):
                    if labels[seedNeighInd] == -1:#NOISE
                        labels[seedNeighInd] = curLabel                      
                    if not labels[seedNeighInd]:#UNCLASSIFIED
                        seeds.append(seedNeighInd)
                        labels[seedNeighInd] = curLabel
                                             
            seeds = seeds[1:]
        return True

class DBSCAN(object):

    def __init__(self, eps=0.5, minSamples=5, metric='euclidean',
                algorithm='auto', nJobs=1):
        '''
        Parameters
        ----------
        eps : float, optional
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.

        minSamples : int, optional
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself.

        metric : string, or callable
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string or callable, it must be one of
            the options allowed by metrics.pairwise.calculate_distance for its
            metric parameter.
     
        algorithm : not implementated now

        nJobs : int, optional (default = 1)
            The number of parallel jobs to run.
            If ``-1``, then the number of jobs is set to the number of CPU cores.

        '''
        self.eps = eps
        self.minSamples = minSamples
        self.metricFn = _euclidean if metric == 'euclidean' else metric
        self.algorithm = algorithm
        self.nJobs = nJobs

    def fit(self,XMat,yVec=None):
        """Perform DBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        XMat : a matrix of data, row means sample space, col means feature space
            
        Outputs:
        labels: -1 means NOISE;
                None means UNCLASSIFIED
        """
        nSamples = XMat.shape[0]
        labels = [None]*nSamples
        curLabel = 1
        for curSampleInd in range(nSamples):
            if not labels[curSampleInd]:
                if _curSample_expand(XMat, labels, self.metricFn,
                                     curSampleInd, curLabel, 
                                     self.eps, self.minSamples):
                    curLabel += 1

        self.labels = labels
        return labels

    def fit_predict(self,XMat,yVec=None):
        pass

if __name__ == "__main__":
    XMat = np.asarray([[1, 1.1],
                       [1.2,0.8],
                       [0.8, 1],
                       [3.7, 4],
                       [3.9,3.9],
                       [3.6,4.1],
                       [10,10]])
    dbscan = DBSCAN(minSamples=2)
    start = time.time()
    labels = dbscan.fit(XMat)
    print(time.time()-start)
    assert labels == [1, 1, 1, 2, 2, 2, -1]
    print(labels)