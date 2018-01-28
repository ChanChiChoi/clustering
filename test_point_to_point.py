import numpy as np
from metrics.point_to_point import weighted_lp_DM_vec,lp_DM_mat,dG_DM_vec,dQ_DM_vec

x = np.ones((1, 2))
y = 2 * np.ones((1, 2))
ans = weighted_lp_DM_vec(x,y,p=2)
print(ans)

X = np.ones((1,2))
Y = 2*np.ones((2,2))
ans = lp_DM_mat(X,Y,p=2)
print(ans)

x = np.array([0,1,2])
y = np.array([4,3,2])
maxVec = np.array([10,12,13])
minVec = np.array([0,0.5,1])
ans = dG_DM_vec(x,y,maxVec,minVec)
print(ans)#0.0922

ans = dQ_DM_vec(x,y)
print(ans)#0.6455