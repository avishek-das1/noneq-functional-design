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
cb='k'

fig = plt.figure(figsize=(36,36))
plt.subplots_adjust(hspace=0.3,wspace=0.2)
gs = gridspec.GridSpec(6, 3)
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),
        fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2]),
        fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1]),fig.add_subplot(gs[2,2]),
        fig.add_subplot(gs[3,0]),fig.add_subplot(gs[3,1]),fig.add_subplot(gs[3,2]),
        fig.add_subplot(gs[4,0]),fig.add_subplot(gs[4,1]),fig.add_subplot(gs[4,2]),
        fig.add_subplot(gs[5,0]),fig.add_subplot(gs[5,1]),fig.add_subplot(gs[5,2])]


x=np.loadtxt('c2v_oh_j/omegatot.txt')
y=np.loadtxt('c2v_oh_j/sheartot.txt')
zn=np.loadtxt('c2v_oh_j/D0runningtot.txt')
z=np.reshape(zn,(int(np.size(zn)/15),15))
axs[0].plot(x[:,1],linestyle='-',lw=lwp,color=ca,label=r'$\tau j_{\mathrm{AB}}$')
axs[1].plot(y,linestyle='-',lw=lwp,color=cb)
for i in range(15):
    axs[2].plot(z,linestyle='-',lw=lwp)
#axs[0].set_xlim(0,70)

x=np.loadtxt('oh_c2v_j/omegatot.txt')
y=np.loadtxt('oh_c2v_j/sheartot.txt')
zn=np.loadtxt('oh_c2v_j/D0runningtot.txt')
z=np.reshape(zn,(int(np.size(zn)/15),15))
axs[3].plot(x[:,1],linestyle='-',lw=lwp,color=ca,label=r'$\tau j_{\mathrm{BA}}$')
axs[4].plot(y,linestyle='-',lw=lwp,color=cb)
for i in range(15):
    axs[5].plot(z,linestyle='-',lw=lwp)
#axs[0].set_xlim(0,70)

x=np.loadtxt('c2v_oh_q_1/omegatot.txt')
y=np.loadtxt('c2v_oh_q_1/sheartot.txt')
zn=np.loadtxt('c2v_oh_q_1/D0runningtot.txt')
z=np.reshape(zn,(int(np.size(zn)/15),15))
axs[6].plot(x[:,1],linestyle='-',lw=lwp,color=ca,label=r'$\tau q_{\mathrm{AB}}$')
axs[7].plot(y,linestyle='-',lw=lwp,color=cb)
for i in range(15):
    axs[8].plot(z,linestyle='-',lw=lwp)
#axs[0].set_xlim(0,70)

x=np.loadtxt('c2v_oh_q_2/omegatot.txt')
y=np.loadtxt('c2v_oh_q_2/sheartot.txt')
zn=np.loadtxt('c2v_oh_q_2/D0runningtot.txt')
z=np.reshape(zn,(int(np.size(zn)/15),15))
axs[12].plot(x[:,1],linestyle='-',lw=lwp,color=ca,label=r'$\tau q_{\mathrm{AB}}$')
axs[13].plot(y,linestyle='-',lw=lwp,color=cb)
for i in range(15):
    axs[14].plot(z,linestyle='-',lw=lwp)
#axs[0].set_xlim(0,70)

x=np.loadtxt('oh_c2v_q_1/omegatot.txt')
y=np.loadtxt('oh_c2v_q_1/sheartot.txt')
zn=np.loadtxt('oh_c2v_q_1/D0runningtot.txt')
z=np.reshape(zn,(int(np.size(zn)/15),15))
axs[9].plot(x[:,1],linestyle='-',lw=lwp,color=ca,label=r'$\tau q_{\mathrm{BA}}$')
axs[10].plot(y,linestyle='-',lw=lwp,color=cb)
for i in range(15):
    axs[11].plot(z,linestyle='-',lw=lwp)
#axs[0].set_xlim(0,70)

x=np.loadtxt('oh_c2v_q_2/omegatot.txt')
y=np.loadtxt('oh_c2v_q_2/sheartot.txt')
zn=np.loadtxt('oh_c2v_q_2/D0runningtot.txt')
z=np.reshape(zn,(int(np.size(zn)/15),15))
axs[15].plot(x[:,1],linestyle='-',lw=lwp,color=ca,label=r'$\tau q_{\mathrm{BA}}$')
axs[16].plot(y,linestyle='-',lw=lwp,color=cb)
for i in range(15):
    axs[17].plot(z,linestyle='-',lw=lwp)
#axs[0].set_xlim(0,70)

for i in range(6):
    axs[i*3].set_yscale('log')
    axs[i*3].set_xlabel(r'$I$')
    axs[i*3+1].set_xlabel(r'$I$')
    axs[i*3+2].set_xlabel(r'$I$')
    axs[i*3].set_ylabel(r'$\langle O\rangle$')
    axs[i*3+1].set_ylabel(r'$f/f^{*}$')
    axs[i*3+2].set_ylabel(r'$D_{ij}/k_{\mathrm{B}}T$')
    leg=axs[i*3].legend(frameon='False',framealpha=1.0,loc='lower right')
    leg.get_frame().set_edgecolor('k')

#axs[3].set_yscale('linear')
#axs[3].set_ylim(-4.5e-6,0.8e-5)
axs[5].set_ylim(0,10)
axs[2].annotate(r'$\mathbf{B}\Rightarrow\Rightarrow\mathbf{C}$',xy=(0.7, 0.7), xycoords='axes fraction',bbox=dict(facecolor='none', edgecolor='k',pad=10))
axs[5].annotate(r'$\mathbf{B}\Rightarrow\Rightarrow\bm{\varnothing}$',xy=(0.7, 0.7), xycoords='axes fraction',bbox=dict(facecolor='none', edgecolor='k',pad=10))
axs[8].annotate(r'$\mathbf{B}\Rightarrow\Rightarrow\mathbf{S}$',xy=(0.7, 0.7), xycoords='axes fraction',bbox=dict(facecolor='none', edgecolor='k',pad=10))
axs[11].annotate(r'$\mathbf{B}\Rightarrow\Rightarrow\mathbf{S}$',xy=(0.7, 0.7), xycoords='axes fraction',bbox=dict(facecolor='none', edgecolor='k',pad=10))
axs[14].annotate(r'$\mathbf{C}\Rightarrow\Rightarrow\mathbf{AS}$',xy=(0.7, 0.7), xycoords='axes fraction',bbox=dict(facecolor='none', edgecolor='k',pad=10))
axs[17].annotate(r'$\mathbf{C}\Rightarrow\Rightarrow\mathbf{S}$',xy=(0.7, 0.7), xycoords='axes fraction',bbox=dict(facecolor='none', edgecolor='k',pad=10))

#axs[0].vlines(196,0,2e-2,color='k',linestyle='--')
#axs[6].vlines(7,0,2e-1,color='k',linestyle='--')
#axs[9].vlines(15,0,2e-2,color='k',linestyle='--')

plt.savefig('sifig1.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()



