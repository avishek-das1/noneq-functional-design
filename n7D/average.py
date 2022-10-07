import numpy as np

x=np.loadtxt('c3v2_2.txt')
y=np.loadtxt('c2v_2.txt')
z=0.5*x+0.5*y
for i in range(21):
    print(z[i])
