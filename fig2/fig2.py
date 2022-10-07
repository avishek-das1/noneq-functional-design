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

fig = plt.figure(figsize=(28,10))
plt.subplots_adjust(hspace=1.6,wspace=0.23)
gs = gridspec.GridSpec(5, 3)
axs=[fig.add_subplot(gs[0:2,0]),fig.add_subplot(gs[0:2,1]),fig.add_subplot(gs[0:2,2]),
        fig.add_subplot(gs[2:,0]),fig.add_subplot(gs[2:,1]),fig.add_subplot(gs[2:,2])]

s=np.arange(0,75,5)
x=np.zeros((3,np.size(s),4))
x[0,:,:]=np.loadtxt('shearsweep/solC/run1/omega.txt')
x[1,:,:]=np.loadtxt('shearsweep/solC/run2/omega.txt')
x[2,:,:]=np.loadtxt('shearsweep/solC/run3/omega.txt')

rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]

y=np.mean(x,axis=0)
ye=np.std(x,axis=0)/3**0.5

yr=np.mean(rs,axis=0)
yre=np.std(rs,axis=0)/3**0.5

axs[3].errorbar(s,y[:,0],yerr=ye[:,0],color=(0.35,0.35,0.35),marker='o',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$Y_{\mathrm{A}}$')
axs[3].errorbar(s,yr[:,0],yerr=yre[:,0],color=(0.35,0.35,0.35),marker='^',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$k_{\mathrm{AB}}\tau$')
axs[3].errorbar(s,y[:,2],yerr=ye[:,2],color=(0.35,0.35,0.35),marker='s',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$q_{\mathrm{AB}}\tau$')
axs[3].errorbar(s,y[:,1],yerr=ye[:,1],color=(1,0.5,0),marker='o',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$Y_{\mathrm{B}}$')
axs[3].errorbar(s,yr[:,1],yerr=yre[:,1],color=(1,0.5,0),marker='^',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$k_{\mathrm{BA}}\tau$')
axs[3].errorbar(s,y[:,3],yerr=ye[:,3],color=(1,0.5,0),marker='s',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$q_{\mathrm{BA}}\tau$')

axs[3].set_yscale('log')
axs[3].set_xlabel(r'$f/f^{*}$')
axs[3].set_ylabel(r'$\langle O\rangle$')
axs[3].set_ylim(0.004,1)

#axs[2].legend(loc = 'center right', ncol = 2, columnspacing=0.5,labelspacing=2,handletextpad = 0.5,
#        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(1.73, 0.6))

#adding image
axs[0].axis('off')
axs[1].axis('off')
x0=0.
y0=-0.2
soh=mpimg.imread('../editedstructures/oh_cpk_3.tga')
sc2v=mpimg.imread('../editedstructures/c2v_cpk_3.tga')
sc2vm1=mpimg.imread('../editedstructures/c2vm1_cpk.tga')
sc2vm1=sc2vm1[150:-10,20:-10,:]
#sc3v1=sc3v1[150:-10,40:-10,:]

imagebox1 = OffsetImage(sc2v, zoom=0.18)
ab1 = AnnotationBbox(imagebox1, (x0+0.09,y0+0.6),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
axs[1].annotate(r'$\mathrm{C}_{\mathrm{2v}}(\mathrm{A})$',xy=(x0+0.02, y0+0.05), xycoords='axes fraction')

imagebox1 = OffsetImage(soh, zoom=0.18)
ab1 = AnnotationBbox(imagebox1, (x0+0.88,y0+0.6),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
#adding text
axs[1].annotate(r'$\mathrm{O}_{\mathrm{h}}(\mathrm{B})$',xy=(x0+0.8, y0+0.05), xycoords='axes fraction')

imagebox1 = OffsetImage(sc2vm1, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0+0.47,y0+0.5),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)

circ = Ellipse((x0+0.47, y0+0.6), 0.5, 1.2, color='lightblue', alpha=0.3, clip_on=False,zorder=4)
axs[1].add_patch(circ)
circ.zorder=99

a1 = mpl.patches.FancyArrowPatch((x0+0.2, y0+0.65), (x0+0.75, y0+0.65),
        connectionstyle="arc3,rad=-.5",color=(0.35,0.35,0.35),
        arrowstyle="simple,head_width=12,head_length=17",
        lw=3,clip_on=False,zorder=11)
axs[1].add_patch(a1)
a1.zorder=101
a1 = mpl.patches.FancyArrowPatch((x0+0.2, y0+0.65), (x0+0.235, y0+0.74),
        connectionstyle="arc3,rad=-0.05",color=(0.35,0.35,0.35),
        arrowstyle="simple,head_width=12,head_length=17",
        lw=3,clip_on=False,zorder=11)
axs[1].add_patch(a1)
a1.zorder=101
a2 = mpl.patches.FancyArrowPatch((x0+0.75, y0+0.55),(x0+0.2, y0+0.55),
        connectionstyle="arc3,rad=-.5",color=(1,0.5,0),
        arrowstyle="simple,head_width=12,head_length=17",
        lw=3,clip_on=False,zorder=11)
axs[1].add_patch(a2)
a2.zorder=101
a2 = mpl.patches.FancyArrowPatch((x0+0.75, y0+0.55),(x0+0.725, y0+0.48),
        connectionstyle="arc3,rad=-0.05",color=(1,0.5,0),
        arrowstyle="simple,head_width=12,head_length=17",
        lw=3,clip_on=False,zorder=11)
axs[1].add_patch(a2)
a2.zorder=101

axs[0].annotate(r'$\mathrm{\textbf{C}}$',xy=(0.96, y0+0.05), xycoords='axes fraction')
nu=axs[1].annotate(r'$\boldsymbol{\varnothing}$',xy=(x0+0.45, y0+0.), xycoords='axes fraction',color='r')
nu.zorder=102
nu2=axs[1].annotate(r'$\mathrm{\textbf{C}}$',xy=(x0+0.45, y0+1.1), xycoords='axes fraction')
nu2.zorder=102

Doct=mpimg.imread('../n6D/oct.png')
Dpolytd=mpimg.imread('../n6D/polytd2.png')
DsolA=mpimg.imread('../n6D/solA2.png')

imagebox1 = OffsetImage(DsolA, zoom=0.25,clip_on=False,zorder=5)
ab1 = AnnotationBbox(imagebox1, (x0-0.25,y0+0.55),frameon=True,pad=0.1,xycoords='axes fraction')
axs[1].add_artist(ab1)
ab1.zorder=100

#x0=0.1
#x,y = np.array([[x0+0.005,x0+1], [12,23]])
#line = Line2D(x, y, lw=3, color='grey', alpha=0.7)
#line.set_clip_on(False)
#axs[3].add_line(line)
#x,y = np.array([[x0+0.8,x0+1.2], [12,23]])
#line = Line2D(x, y, lw=3, color='grey', alpha=0.7)
#line.set_clip_on(False)
#axs[3].add_line(line)

#axs[3].arrow(x0+1,23,0.2,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
#axs[3].arrow(x0+1,21.5,0.2-0.4*1.5/11,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
#axs[3].arrow(x0+1,20,0.2-0.4*3/11,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
#axs[3].arrow(x0+1,15,-0.2+0.4*3/11,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
#axs[3].arrow(x0+1,13.5,-0.2+0.4*1.5/11,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
#axs[3].arrow(x0+1,12,-0.2,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)

x=np.loadtxt('displacement/eq/fort.392')
y=np.loadtxt('displacement/neq/fort.392')
l=np.size(x)
ts=np.arange(l)*5e-5*50
axs[2].plot(ts,x,color='brown',lw=2,label='$f/f^{*}=0$')
axs[2].plot(ts,y,color='green',lw=2,label='$f/f^{*}=50$')
axs[2].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
axs[2].set_xlabel(r'$t/t^{*}$')
axs[2].set_ylabel(r'$J_{\mathrm{AB}}$')
axs[2].legend(loc = 'upper left', ncol = 1, columnspacing=0.5,labelspacing=0.5,handletextpad = 0.5,
        borderpad=0.0,frameon='False',framealpha=0.0)#,bbox_to_anchor=(0.73, 0.6))
axs[2].set_xlim(0,50)
axs[2].set_ylim(-0.3e4,2e4)

x=np.loadtxt('shear_covm_cove.txt')
axs[5].errorbar(x[:,0],x[:,1],yerr=x[:,2],color='dodgerblue',marker='x',linestyle='None',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5)
axs[5].set_xlabel(r'$f/f^{*}$')
#axs[5].set_ylabel(r'$\Delta Cov(W,\tau_\mathrm{I})$')
axs[5].set_ylabel(r'$\chi$')

#axins = inset_axes(axs[3], width='100%',height='100%',bbox_to_anchor=(720,400,250,200))
#x=np.loadtxt('duration11/bondlistBA.txt')
#l=np.shape(x)[0]
##print(l)
#xh,xb=np.histogram(x,bins=np.arange(8.5,12.5,1),density=False)
#bs=np.arange(9,12,1)
#xh=xh*5e-5/3165
#axins.bar(bs,xh,color=(1,0.5,0),alpha=0.8)

#x=np.loadtxt('duration11/bondlistAB.txt')
#l=np.shape(x)[0]
##print(l)
#xh,xb=np.histogram(x,bins=np.arange(8.5,12.5,1),density=False)
#bs=np.arange(9,12,1)+0.1
#xh=xh*5e-5/2891
#axins.bar(bs,xh,color=(0.35,0.35,0.35),alpha=0.8)

#axins.set_xlabel(r'$b$')
#axins.set_ylabel(r'$\langle\tau^{(b)}\rangle$')
#axins.set_ylim(0,0.007)
#axins.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

xn=np.loadtxt('duration11/bond11AB.txt')
l1=np.shape(xn)[0]
yn=np.loadtxt('duration11/bond11BA.txt')
l2=np.shape(yn)[0]

xp,xq=np.histogram(xn,bins=20,range=(0.001,0.025),density=True)
xqq=0.5*(xq[:-1]+xq[1:])
yp,yq=np.histogram(yn,bins=20,range=(0.001,0.025),density=True)
yqq=0.5*(yq[:-1]+yq[1:])

axs[4].plot(xqq,xp,color=(0.35,0.35,0.35),marker='v',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,label=r'$P(\tau_\mathrm{I}^{\mathrm{AB}})$')
axs[4].plot(yqq,yp,color=(1,0.5,0),marker='v',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,label=r'$P(\tau_\mathrm{I}^{\mathrm{BA}})$')
axs[4].set_xlabel(r'$\tau_\mathrm{I}$')
axs[4].set_ylabel(r'$P(\tau_\mathrm{I})$')
axs[4].set_xlim(0.001,0.015)
axs[4].set_ylim(0,180)
axs[4].legend(loc = 'upper right', ncol = 1, columnspacing=2.,
        labelspacing=0.5,handletextpad = 0.5,
        borderpad=0.0,frameon='False',framealpha=0.0)#,bbox_to_anchor=(0.6, 1))

h1, l1 = axs[3].get_legend_handles_labels()
#h2, l2 = axs[4].get_legend_handles_labels()
axs[0].legend(h1, l1, loc = 'lower center', ncol = 2, columnspacing=1.4,
        labelspacing=1,handletextpad = 0.5,
        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(0.3, -0.25))

plt.savefig('fig2.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()

#np.savetxt('fig1_wca.txt',np.stack((x,FAA),axis=1))
#np.savetxt('fig1_wca_morse.txt',np.stack((x,FAA+VmorseA),axis=1))


