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

#mpl.style.use("classic")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams.update({'font.size': 32})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,bm}')

fig = plt.figure(figsize=(14,7))
plt.subplots_adjust(hspace=0.7,wspace=0.23)
#gs = gridspec.GridSpec(2, 1)
#axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0])]

gs = gridspec.GridSpec(3, 8)
axs=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,2:6]),fig.add_subplot(gs[2,3:5])]

x=np.loadtxt('paramsweep/c2v_S_oh_AS_c2v/omega.txt')
xn=np.reshape(x,(3,41,4))
xnm=np.mean(xn,axis=0)
xne=np.std(xn,axis=0)/3**0.5
ps=np.arange(41)
axs[0].plot(ps,xnm[:,2],color=(0.35,0.35,0.35),marker='None',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,label=r'$q_{\mathrm{AB}}\tau$')
axs[0].plot(ps,xnm[:,3],color=(1,0.5,0),marker='None',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,label=r'$q_{\mathrm{BA}}\tau$')
axs[0].xaxis.set_ticks([])
#axs[0].set_yscale('log')
axs[0].set_xlim(0,40)
axs[0].set_ylim(0,0.04)
axs[0].yaxis.set_ticks([0,0.02,0.04])
axs[0].vlines(10,0,0.04,lw=2,linestyle=':',color='k',alpha=1.0)
axs[0].vlines(20,0,0.04,lw=2,linestyle=':',color='k',alpha=1.0)
axs[0].vlines(30,0,0.04,lw=2,linestyle=':',color='k',alpha=1.0)
axs[0].set_ylabel(r'$q\tau$')
axs[0].annotate(r'$\mathrm{\textbf{A}}$',xy=(-0.02, -0.3), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{\textbf{S}}$',xy=(0.24, -0.3), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{\textbf{B}}$',xy=(0.49, -0.3), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{\textbf{AS}}$',xy=(0.72, -0.3), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{\textbf{A}}$',xy=(0.98, -0.3), xycoords='axes fraction')

x=np.loadtxt('paramsweep/S_AS_C_null/omega.txt')
xn=np.reshape(x,(3,31,4))
xnm=np.mean(xn,axis=0)
xne=np.std(xn,axis=0)/3**0.5
ps=np.arange(31)
axs[1].plot(ps[:21],xnm[:21,2],color=(0.35,0.35,0.35),marker='None',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2)
axs[1].plot(ps[:21],xnm[:21,3],color=(1,0.5,0),marker='None',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2)
axs[1].xaxis.set_ticks([])
#axs[1].set_yscale('log')
axs[1].set_xlim(0,20)
axs[1].set_ylim(0.015,0.04)
axs[1].vlines(10,0,0.04,lw=2,linestyle=':',color='k',alpha=1.0)
axs[1].set_ylabel(r'$q\tau$')
axs[1].annotate(r'$\mathrm{\textbf{S}}$',xy=(-0.02, -0.3), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{\textbf{AS}}$',xy=(0.47, -0.3), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{\textbf{C}}$',xy=(0.98, -0.3), xycoords='axes fraction')

xne=np.std(xn,axis=0)/3**0.5
ps=np.arange(11)
axs[2].plot(ps[:],xnm[20:,2],color=(0.35,0.35,0.35),marker='None',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2)
axs[2].plot(ps[:],xnm[20:,3],color=(1,0.5,0),marker='None',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2)
axs[2].xaxis.set_ticks([])
#axs[1].set_yscale('log')
axs[2].set_xlim(0,10)
#axs[2].set_ylim(0.015,0.04)
#axs[1].vlines(10,0,0.04,lw=2,linestyle=':',color='k',alpha=1.0)
axs[2].set_ylabel(r'$q\tau$')
axs[2].annotate(r'$\mathrm{\textbf{C}}$',xy=(-0.02, -0.3), xycoords='axes fraction')
#axs[1].annotate(r'$\mathrm{\textbf{AS}}$',xy=(0.47, -0.3), xycoords='axes fraction')
axs[2].annotate(r'$\mathrm{\bm{\varnothing}}$',xy=(0.98, -0.3), xycoords='axes fraction')


#axs[0].annotate(r'$\mathrm{a)}$',xy=(0.0, 0.75), xycoords='figure fraction')

#axs[0].annotate(r'$\mathrm{b)}$',xy=(0.0, 0.52), xycoords='figure fraction')

axs[0].legend(loc = 'upper center', ncol = 2, columnspacing=2.5,labelspacing=0.5,handletextpad = 0.5,
        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(0.5, 1.7))

plt.savefig('sifig2.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()



