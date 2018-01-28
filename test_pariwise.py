
import numpy as np
from metrics.pariwise import euclidean_distances

X = np.random.randn(2,2)
distances = euclidean_distances(X,X)
print(distances)