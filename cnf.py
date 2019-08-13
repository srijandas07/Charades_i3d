import numpy as np
#cnf / cnf.astype(np.float).sum(axis=1)

cnf=np.load('i3d_CS.npy')

num=32
sum=0                                     
cnf_1=cnf / cnf.astype(np.float).sum(axis=1)
for i in range(num):
    sum=sum+cnf_1[i][i]
sum/num
