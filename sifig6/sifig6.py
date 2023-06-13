import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.patches import Rectangle,Ellipse
import matplotlib.colors as mcol
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit

def func(x, a, b):
  return a + b * x**2

#mpl.style.use("classic")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams.update({'font.size': 32})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,bm}')

fig = plt.figure(figsize=(14,12))
plt.subplots_adjust(hspace=0.3,wspace=0.3)
#gs = gridspec.GridSpec(2, 1)
#axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0])]

gs = gridspec.GridSpec(2, 2)
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]

m=mpimg.imread('image1.png')
#print(np.shape(m))
imagebox1 = OffsetImage(m[100:440,250:730,:], zoom=0.8)
ab1 = AnnotationBbox(imagebox1, (0.35,0.5),frameon=False,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].axis('off')

mu=np.loadtxt('pathwayfrac_mu.txt')
nu=np.loadtxt('pathwayfrac_nu.txt')
axs[1].errorbar(mu[:,0],mu[:,1],yerr=mu[:,2],color='orange',marker='o',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$p_{\mu}$')
axs[1].errorbar(nu[:,0],nu[:,1],yerr=nu[:,2],color='green',marker='o',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$p_{\nu}$')
#axs[0].xaxis.set_ticks([])
#axs[0].xaxis.set_ticks([])
#axs[0].set_xlim(0,40)
axs[1].set_ylim(0,1)
axs[1].set_ylabel(r'$p_{\omega\in\{\mu,\nu\}}$')
axs[1].set_xlabel(r'$f/f^{*}$')

axs[1].legend(loc = 'upper center', ncol = 2, columnspacing=2.5,labelspacing=0.5,handletextpad = 0.5,
        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(0.5, 0.6))

xal=np.loadtxt('anisotropy.txt')
axs[2].errorbar(xal[:,0],xal[:,1],yerr=xal[:,2],color='#93bc81ff',marker='s',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5)
axs[2].set_xlabel(r'$f/f^{*}$')
axs[2].set_ylabel(r'$\langle \delta x_{r}\delta z_{r}\rangle/\sigma^{2}$')

Rx=np.loadtxt('Rx.txt')
Ry=np.loadtxt('Ry.txt')
Rz=np.loadtxt('Rz.txt')
axs[3].errorbar(Rx[:,0],Rx[:,1]/2,yerr=Rx[:,2]/2,color='r',marker='^',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$R_{x}$')
axs[3].errorbar(Ry[:,0],Ry[:,1]/2,yerr=Ry[:,2]/2,color='b',marker='^',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$R_{y}$')
axs[3].errorbar(Rz[:,0],Rz[:,1]/2,yerr=Rz[:,2]/2,color='k',marker='^',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$R_{z}$')
axs[3].set_xlabel(r'$f/f^{*}$')
axs[3].set_ylabel(r'$R\gamma/k_{\mathrm{B}}T$')

popt, _ = curve_fit(func, Rx[:4,0], Rx[:4,1]/2)
xnew = np.linspace(0, 4, 100)
axs[3].plot(xnew, func(xnew, *popt),marker='None',lw=2,color='r',linestyle='--')

axs[3].legend(loc = 'upper left', ncol = 1, columnspacing=2.5,labelspacing=0.5,handletextpad = 0.5,
        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(0., 1.))

axs[0].annotate(r'$\nu$',xy=(0.04, 0.73), xycoords='axes fraction',zorder=100)
axs[0].annotate(r'$\mu$',xy=(0.04, 0.23), xycoords='axes fraction',zorder=100)

axs[0].annotate(r'$\mathrm{(a)}$',xy=(-0.25, 1.0), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{(b)}$',xy=(-0.28, 1.0), xycoords='axes fraction')
axs[2].annotate(r'$\mathrm{(c)}$',xy=(-0.25, 1.0), xycoords='axes fraction')
axs[3].annotate(r'$\mathrm{(d)}$',xy=(-0.28, 1.0), xycoords='axes fraction')

plt.savefig('sifig6.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()



