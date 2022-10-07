import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.tri as tri
from numba import jit

#mpl.style.use("classic")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams.update({'font.size': 22})
plt.rc('text', usetex=True)

@jit
def pot(x,D,ye,yk):
    V=0
    urc=D*(np.exp(-2.*10*1)-2.*np.exp(-10*1))
    frc=2*10*D*(np.exp(-2.*10*1)-np.exp(-10*1))
    urcY=ye*np.exp(-yk*5)/5
    frcY=ye*np.exp(-yk*5)/5**2 *(1+yk*5)
    if x<2**(1/6):
        V+=4.*10*(1/x**12-1/x**6)+10
    if x<2**(1/6)+1:
        V+=D*(np.exp(-2.*10*(x-2**(1/6)))-2.*np.exp(-10*(x-2**(1/6))))++(x-(2**(1/6)+1))*frc-urc
    if x<=5:
        V+=ye*np.exp(-yk*x)/x+(x-5)*frcY-urcY
    return V

@jit
def hillbarrier(D,epsilon,kappa):
    xs=np.linspace(1.1,5,100)
    V=np.zeros(100)
    for j in range(100):
        V[j]=pot(xs[j],D,epsilon,kappa)
    b=np.amax(V)-np.amin(V)
    c=np.amax(V)-V[-1]
    d=np.amin(V)-V[-1]
    return c,b #hill and barrier

Ds=np.loadtxt('D0running.txt')
Ys=np.loadtxt('Yekrunning.txt')

l=np.size(Ds)
cb=np.zeros((l,2))
for i in range(l):
    cb[i,0],cb[i,1]=hillbarrier(Ds[i],Ys[i,0],Ys[i,1])

ngridx = 20
ngridy = 20
npts=np.size(Ds)

xi = np.linspace(0, 3.2, ngridx)
yi = np.linspace(2, 7.5, ngridy)

x=cb[:,0]
y=cb[:,1]
z=(np.loadtxt('omega.txt'))[:,0]
fig, ax2 = plt.subplots(nrows=1,ncols=1)

#print(xi)
#print(yi)
#triang = tri.Triangulation(xi, yi)
#interpolator = tri.LinearTriInterpolator(triang, z)
#Xi, Yi = np.meshgrid(xi, yi)
#zi = interpolator(Xi, Yi)

#ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
#cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

#fig.colorbar(cntr1, ax=ax1)
#ax1.plot(x, y, 'ko', ms=3)
#ax1.set(xlim=(-2, 2), ylim=(-2, 2))
#ax1.set_title('grid and contour (%d points, %d grid points)' %
#              (npts, ngridx * ngridy))

# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

ax2.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
cntr2 = ax2.tricontourf(x, y, z, levels=14, cmap="viridis")

fig.colorbar(cntr2, ax=ax2)
ax2.plot(x, y, 'ro', ms=3)
ax2.set(xlim=(0, 3.2), ylim=(2, 7.5))
ax2.set_title('tricontour (%d points)' % npts)

plt.subplots_adjust(hspace=0.5)
plt.show()


#plt.xlim(0,3.2)
#plt.ylim(2,7.5)
#print(outcount,scount,s2count,(M1*M2*M3))
#plt.show()

#plt.plot(hb4[i,0],hb4[i,1],color=m.to_rgba(hb4[i,2]),marker='o',markersize=5,alpha=0.2)
#plt.colorbar(m)
#plt.tight_layout()
#plt.xlabel(r'Hill')
#plt.ylabel(r'Barrier')
#plt.show()

