import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

P=6
N=720
M=15

a=np.loadtxt('solC2.txt')
b=np.loadtxt('test.txt')
an=np.zeros((P,P))
bn=np.zeros((P,P))
icount=0
for i in range(P):
    for j in range(i+1,P):
        an[i,j]=a[icount]
        an[j,i]=an[i,j]
        bn[i,j]=b[icount]
        bn[j,i]=bn[i,j]
        icount+=1

x=np.arange(P) #particle labels
perms=permutations(x) #kth permutation, a[n] is b[perms[k,n]]

err=np.zeros(N)
icount=0
plist=np.zeros((N,6))
for p in perms:
    plist[icount,:]=p
    for i in range(P):
        for j in range(i+1,P):
            err[icount]+=100*(an[i,j]-bn[p[i],p[j]])**2+(i-p[i])**2+(j-p[j])**2
    icount+=1
m=np.argmin(err)

#AD
print(plist[m,:])
#plt.plot(err,'ko')
#plt.show()

newbn=np.zeros((P,P))
newb=np.zeros(M)
for i in range(P):
    for j in range(P):
        newbn[i,j]=bn[int(plist[m,i]),int(plist[m,j])]
icount=0
for i in range(P):
    for j in range(i+1,P):
        newb[icount]=newbn[i,j]
        icount+=1
np.savetxt('test2.txt',newb)




