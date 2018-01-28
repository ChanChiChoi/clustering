import numpy as np
from metrics.point_to_point import weighted_lp_DM_vec,lp_DM_mat

x = np.ones((1, 2))
y = 2 * np.ones((1, 2))
ans = weighted_lp_DM_vec(x,y,p=2)
print(ans)

X = np.ones((1,2))
Y = 2*np.ones((2,2))
ans = lp_DM_mat(X,Y,p=2)
print(ans)