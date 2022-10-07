import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.colors as mcol
import matplotlib.cm as cm

#mpl.style.use("classic")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams.update({'font.size': 32})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,bm}')

lwp=1.5
ca='r'
cb='b'
cc='k'
cd='orange'

fig = plt.figure(figsize=(36,24))
plt.subplots_adjust(hspace=0.3,wspace=0.3)
gs = gridspec.GridSpec(4, 4)
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[0,3]),
        fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2]),fig.add_subplot(gs[1,3]),
        fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1]),fig.add_subplot(gs[2,2]),fig.add_subplot(gs[2,3]),
        fig.add_subplot(gs[3,0]),fig.add_subplot(gs[3,1]),fig.add_subplot(gs[3,2]),fig.add_subplot(gs[3,3])]

N=90

x=np.loadtxt('den_1/omega.txt')
y=np.loadtxt('den_1/D0running.txt')
z=np.loadtxt('den_1/Yekrunning.txt')
axs[0].plot(x[:,1]/N,linestyle='-',lw=lwp,color=ca,label=r'$Y_{6}$')
axs[1].plot(y,linestyle='-',lw=lwp,color=cb)
axs[2].plot(z[:,0],linestyle='-',lw=lwp,color=cc)
axs[3].plot(z[:,1],linestyle='-',lw=lwp,color=cd)

x=np.loadtxt('den_25/omega.txt')
y=np.loadtxt('den_25/D0running.txt')
z=np.loadtxt('den_25/Yekrunning.txt')
axs[4].plot(x[:,1]/N,linestyle='-',lw=lwp,color=ca,label=r'$Y_{6}$')
axs[5].plot(y,linestyle='-',lw=lwp,color=cb)
axs[6].plot(z[:,0],linestyle='-',lw=lwp,color=cc)
axs[7].plot(z[:,1],linestyle='-',lw=lwp,color=cd)

x=np.loadtxt('f500_1/omega.txt')
y=np.loadtxt('f500_1/D0running.txt')
z=np.loadtxt('f500_1/Yekrunning.txt')
axs[8].plot(x[:,1]/N,linestyle='-',lw=lwp,color=ca,label=r'$q_{6}\tau$')
axs[9].plot(y,linestyle='-',lw=lwp,color=cb)
axs[10].plot(z[:,0],linestyle='-',lw=lwp,color=cc)
axs[11].plot(z[:,1],linestyle='-',lw=lwp,color=cd)

x=np.loadtxt('f500_25/omega.txt')
y=np.loadtxt('f500_25/D0running.txt')
z=np.loadtxt('f500_25/Yekrunning.txt')
axs[12].plot(x[:,1]/N,linestyle='-',lw=lwp,color=ca,label=r'$q_{6}\tau$')
axs[13].plot(y,linestyle='-',lw=lwp,color=cb)
axs[14].plot(z[:,0],linestyle='-',lw=lwp,color=cc)
axs[15].plot(z[:,1],linestyle='-',lw=lwp,color=cd)

for i in range(4):
#    axs[i*4].set_yscale('log')
    axs[i*4].set_xlabel(r'$I$')
    axs[i*4+1].set_xlabel(r'$I$')
    axs[i*4+2].set_xlabel(r'$I$')
    axs[i*4+3].set_xlabel(r'$I$')
    axs[i*4].set_ylabel(r'$\langle O\rangle$')
    axs[i*4+1].set_ylabel(r'$D/k_{\mathrm{B}}T$')
    axs[i*4+2].set_ylabel(r'$\epsilon/k_{\mathrm{B}}T$')
    axs[i*4+3].set_ylabel(r'$\kappa\sigma$')

axs[8].set_yscale('log')
axs[12].set_yscale('log')
#axs[0].set_ylim(-0.02,0.235)
axs[4].set_ylim(0.03,0.18)

fa=0.0
leg=axs[0].legend(frameon='False',framealpha=fa,loc='upper left')
leg.get_frame().set_edgecolor('k')
leg=axs[4].legend(frameon='False',framealpha=fa,loc='upper left')
leg.get_frame().set_edgecolor('k')
leg=axs[8].legend(frameon='False',framealpha=fa,loc='lower right')
leg.get_frame().set_edgecolor('k')
leg=axs[12].legend(frameon='False',framealpha=fa,loc='lower right')
leg.get_frame().set_edgecolor('k')

plt.savefig('sifig4.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()



