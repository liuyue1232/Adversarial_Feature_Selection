import numpy as np


Z = [[1,0,1,0],[0,1,0,1]]

d = 4

hatPF=np.mean(Z,axis=0)
kbar=np.sum(hatPF)
denom=(kbar/d)*(1-kbar/d)


print(hatPF)
print(kbar)
print(denom)