import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams.update({'font.size': 22})
plt.rc('text', usetex=True)

D0AA=np.loadtxt('c3v1_c2_f_symm2.txt')
D0=np.zeros((7,7))
mycount=0
for i in range(7):
    for j in range(i+1,7):
        D0[i,j]=D0AA[mycount]
        D0[j,i]=D0[i,j]
        mycount+=1
plt.matshow(D0,fignum=0,vmin=0,vmax=10,cmap='Blues')
plt.axis('off')
plt.savefig("c3v1_c2_f_symm2.png", bbox_inches='tight')
#plt.show()

