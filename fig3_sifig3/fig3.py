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

fig = plt.figure(figsize=(8,26))
plt.subplots_adjust(hspace=0.1,wspace=0.27)
gs = gridspec.GridSpec(3, 1)
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[2,0])]

#axs[0].legend(loc = 'upper center', ncol = 1, columnspacing=1.7,handletextpad = 1.0,
#        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(0.58, 1.0))

s=np.arange(0,75,5)
x=np.zeros((3,np.size(s),4))

z=np.loadtxt('shearsweep/symms/c2_c2v_f2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[0].plot(s,y[:,2],color='green',linestyle='-',lw=3)
axs[0].plot(s,y[:,3],color='green',linestyle='--',lw=3)

z=np.loadtxt('shearsweep/symms/c3v1_c2_f_symm2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[0].plot(s,y[:,2],color='blueviolet',linestyle='--',lw=3)
axs[0].plot(s,y[:,3],color='blueviolet',linestyle='-',lw=3)

z=np.loadtxt('shearsweep/symms/c3v1_c2v_f_st2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[0].plot(s,y[:,2],color='deeppink',linestyle='-',lw=3)
axs[0].plot(s,y[:,3],color='deeppink',linestyle='--',lw=3)

z=np.loadtxt('shearsweep/symms/c3v1_c3v2_f2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[0].plot(s,y[:,2],color='brown',linestyle='-',lw=3)
axs[0].plot(s,y[:,3],color='brown',linestyle='--',lw=3)

z=np.loadtxt('shearsweep/symms/c3v2_c2_f2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[0].plot(s,y[:,2],color='lime',linestyle='-',lw=3)
axs[0].plot(s,y[:,3],color='lime',linestyle='--',lw=3)

axs[0].set_yscale('log')
axs[0].set_xlim(0,70)
#axs[0].set_xlabel(r'$f/f^{*}$')
axs[0].set_ylabel(r'$q\tau$')

z=np.loadtxt('shearsweep/asymms/c2_c3v1_f2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[1].plot(s,y[:,2],color='blueviolet',linestyle='--',lw=3)
axs[1].plot(s,y[:,3],color='blueviolet',linestyle='-',lw=3)

z=np.loadtxt('shearsweep/asymms/c2_c3v2_f2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[1].plot(s,y[:,2],color='lime',linestyle='--',lw=3)
axs[1].plot(s,y[:,3],color='lime',linestyle='-',lw=3)

z=np.loadtxt('shearsweep/asymms/c2v_c2_f2/omega.txt')
#z=np.loadtxt('shearsweep/asymms/c2v_c2_f_startfromC2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[1].plot(s,y[:,2],color='green',linestyle='-',lw=3)
axs[1].plot(s,y[:,3],color='green',linestyle='--',lw=3)

z=np.loadtxt('shearsweep/asymms/c2v_c3v2_f2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[1].plot(s,y[:,2],color='k',linestyle='--',lw=3)
axs[1].plot(s,y[:,3],color='k',linestyle='-',lw=3)

z=np.loadtxt('shearsweep/asymms/c3v1_c2v_f2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[1].plot(s,y[:,2],color='deeppink',linestyle='-',lw=3)
axs[1].plot(s,y[:,3],color='deeppink',linestyle='--',lw=3)

z=np.loadtxt('shearsweep/asymms/c3v2_c2v_f2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[1].plot(s,y[:,2],color='dodgerblue',linestyle='-',lw=3)
axs[1].plot(s,y[:,3],color='dodgerblue',linestyle='--',lw=3)

z=np.loadtxt('shearsweep/asymms/c2_c2v_f_asymm2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[1].plot(s,y[:,2],color='orange',linestyle='--',lw=3)
axs[1].plot(s,y[:,3],color='orange',linestyle='-',lw=3)

axs[1].set_yscale('log')
axs[1].set_xlim(0,70)
#axs[1].set_xlabel(r'$f/f^{*}$')
axs[1].set_ylabel(r'$q\tau$')

z=np.loadtxt('shearsweep/curr/c2_c2v_c2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[2].plot(s,y[:,2],color='orange',linestyle='--',lw=3)
axs[2].plot(s,y[:,3],color='orange',linestyle='-',lw=3)

z=np.loadtxt('shearsweep/curr/c2_c3v1_c2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[2].plot(s,y[:,2],color='blueviolet',linestyle='--',lw=3)
axs[2].plot(s,y[:,3],color='blueviolet',linestyle='-',lw=3)

z=np.loadtxt('shearsweep/curr/c2_c3v2_c2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[2].plot(s,y[:,2],color='lime',linestyle='--',lw=3)
axs[2].plot(s,y[:,3],color='lime',linestyle='-',lw=3)

z=np.loadtxt('shearsweep/curr/c2v_c2_c2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[2].plot(s,y[:,2],color='green',linestyle='-',lw=3)
axs[2].plot(s,y[:,3],color='green',linestyle='--',lw=3)

z=np.loadtxt('shearsweep/curr/c3v1_c2v_c2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[2].plot(s,y[:,2],color='deeppink',linestyle='-',lw=3)
axs[2].plot(s,y[:,3],color='deeppink',linestyle='--',lw=3)

z=np.loadtxt('shearsweep/curr/c3v2_c2v_c2/omega.txt')
x=np.reshape(z,(3,np.size(s),4))
rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]
y=np.mean(x,axis=0)
axs[2].plot(s,y[:,2],color='dodgerblue',linestyle='-',lw=3)
axs[2].plot(s,y[:,3],color='dodgerblue',linestyle='--',lw=3)

axs[2].set_yscale('log')
axs[2].set_xlim(0,70)
axs[2].set_xlabel(r'$f/f^{*}$')
axs[2].set_ylabel(r'$q\tau$')

axs[0].get_xaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)

#axs[1].annotate(r'$\mathrm{\textbf{S}}$',xy=(68,0.61),xycoords='data',color='k')
#axs[2].annotate(r'$\mathrm{\textbf{AS}}$',xy=(64,0.61),xycoords='data',color='k')

#adding image
#axs[0].axis('off')
sc3v1=mpimg.imread('../editedstructures/c3v1_cpk.tga')
sc3v1=sc3v1[150:-360,40:-10,:]
sc3v2=mpimg.imread('../editedstructures/c3v2_cpk.tga')
sc3v2=sc3v2[200:-100,40:-10,:]
sc2=mpimg.imread('../editedstructures/c2_cpk.tga')
sc2=sc2[230:-100,15:-10,:]
sc2v=mpimg.imread('../editedstructures/n7c2v_cpk.tga')
sc2v=sc2v[360:-420,70:-10,:]

x0=1.2
y0=-1.1
imagebox1 = OffsetImage(sc3v1, zoom=0.27)
ab1 = AnnotationBbox(imagebox1, (x0+0.1,y0+1.7),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc3v2, zoom=0.25,zorder=2)
ab1 = AnnotationBbox(imagebox1, (x0+0.8,y0+1.6),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (x0+0.45,y0+1.9),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2v, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (x0+0.45,y0+1.3),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
ab1.zorder=1

#adding text
axs[0].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(x0+-0.13, y0+1.6), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(x0+0.9, y0+1.6), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{C}_{\mathrm{2}}$',xy=(x0+0.26, y0+2), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{C}_{\mathrm{2v}}$',xy=(x0+0.59, y0+1.25), xycoords='axes fraction')

axs[0].annotate("", xy=(x0+0.38, y0+1.92), xytext=(x0+0.20, y0+1.71),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))
axs[0].annotate("", xy=(x0+0.36, y0+1.96), xytext=(x0+0.18, y0+1.75),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='--'))
axs[0].annotate("", xy=(x0+0.25, y0+1.65), xytext=(x0+0.68, y0+1.65),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='brown',linestyle='-'))
axs[0].annotate("", xy=(x0+0.25, y0+1.6), xytext=(x0+0.68, y0+1.6),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='brown',linestyle='--'))
axs[0].annotate("", xy=(x0+0.45, y0+1.45), xytext=(x0+0.45, y0+1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='--'))
axs[0].annotate("", xy=(x0+0.49, y0+1.45), xytext=(x0+0.49, y0+1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))
axs[0].annotate("", xy=(x0+0.15, y0+1.49), xytext=(x0+0.32, y0+1.3),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='--'))
axs[0].annotate("", xy=(x0+0.18, y0+1.51), xytext=(x0+0.35, y0+1.32),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))
axs[0].annotate("", xy=(x0+0.72, y0+1.71), xytext=(x0+0.54, y0+1.91),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))
axs[0].annotate("", xy=(x0+0.73, y0+1.76), xytext=(x0+0.55, y0+1.96),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='--'))

x0=1.2
y0=-1.1
imagebox1 = OffsetImage(sc3v1, zoom=0.27)
ab1 = AnnotationBbox(imagebox1, (x0+0.1,y0+1.7),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc3v2, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0+0.8,y0+1.6),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (x0+0.45,y0+1.9),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2v, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (x0+0.45,y0+1.3),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
ab1.zorder=1

#adding text
axs[1].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(x0+-0.13, y0+1.6), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(x0+0.9, y0+1.6), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{C}_{\mathrm{2}}$',xy=(x0+0.26, y0+2.), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{C}_{\mathrm{2v}}$',xy=(x0+0.58, y0+1.22), xycoords='axes fraction')

axs[1].annotate("", xy=(x0+0.37, y0+1.94), xytext=(x0+0.19, y0+1.73),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))
axs[1].annotate("", xy=(x0+0.44, y0+1.45), xytext=(x0+0.44, y0+1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='orange',linestyle='-'))
axs[1].annotate("", xy=(x0+0.49, y0+1.45), xytext=(x0+0.49, y0+1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))
axs[1].annotate("", xy=(x0+0.165, y0+1.5), xytext=(x0+0.335, y0+1.31),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))
axs[1].annotate("", xy=(x0+0.725, y0+1.735), xytext=(x0+0.545, y0+1.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))
axs[1].annotate("", xy=(x0+0.72, y0+1.51), xytext=(x0+0.555, y0+1.34),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='dodgerblue',linestyle='-'))
axs[1].annotate("", xy=(x0+0.76, y0+1.49), xytext=(x0+0.585, y0+1.31),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='k',linestyle='-'))

x0=1.2
y0=-1.1
imagebox1 = OffsetImage(sc3v1, zoom=0.27)
ab1 = AnnotationBbox(imagebox1, (x0+0.1,y0+1.7),frameon=False,xycoords='axes fraction')
axs[2].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc3v2, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0+0.8,y0+1.6),frameon=False,xycoords='axes fraction')
axs[2].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (x0+0.45,y0+1.9),frameon=False,xycoords='axes fraction')
axs[2].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2v, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (x0+0.45,y0+1.3),frameon=False,xycoords='axes fraction')
axs[2].add_artist(ab1)
ab1.zorder=1

#adding text
axs[2].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(x0+-0.13, y0+1.6), xycoords='axes fraction')
axs[2].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(x0+0.9, y0+1.6), xycoords='axes fraction')
axs[2].annotate(r'$\mathrm{C}_{\mathrm{2}}$',xy=(x0+0.26, y0+2.), xycoords='axes fraction')
axs[2].annotate(r'$\mathrm{C}_{\mathrm{2v}}$',xy=(x0+0.58, y0+1.22), xycoords='axes fraction')

axs[2].annotate("", xy=(x0+0.37, y0+1.94), xytext=(x0+0.19, y0+1.73),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))
axs[2].annotate("", xy=(x0+0.37, y0+1.94), xytext=(x0+0.37+(0.19-0.37)/10, y0+1.94+(1.73-1.94)/10),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))

axs[2].annotate("", xy=(x0+0.44, y0+1.45), xytext=(x0+0.44, y0+1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='orange',linestyle='-'))
axs[2].annotate("", xy=(x0+0.44, y0+1.85+(1.45-1.85)/10), xytext=(x0+0.44, y0+1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='orange',linestyle='-'))

axs[2].annotate("", xy=(x0+0.49, y0+1.45), xytext=(x0+0.49, y0+1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))
axs[2].annotate("", xy=(x0+0.49, y0+1.45), xytext=(x0+0.49, y0+1.45+(1.85-1.45)/10),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))

axs[2].annotate("", xy=(x0+0.165, y0+1.5), xytext=(x0+0.335, y0+1.31),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))
axs[2].annotate("", xy=(x0+0.165, y0+1.5), xytext=(x0+0.165+(0.335-0.165)/10, y0+1.5+(1.31-1.5)/10),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))

axs[2].annotate("", xy=(x0+0.725, y0+1.735), xytext=(x0+0.545, y0+1.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))
axs[2].annotate("", xy=(x0+0.545+(0.725-0.545)/10, y0+1.935+(1.735-1.935)/10), xytext=(x0+0.545, y0+1.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))

axs[2].annotate("", xy=(x0+0.74, y0+1.5), xytext=(x0+0.57, y0+1.325),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='dodgerblue',linestyle='-'))
axs[2].annotate("", xy=(x0+0.74, y0+1.5), xytext=(x0+0.74+(0.57-0.74)/10, y0+1.5+(1.325-1.5)/10),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='dodgerblue',linestyle='-'))

#axs[0].arrow(1.93,19,0.5,0,lw=3,color=(0.35,0.35,0.35),head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
#axs[0].arrow(2.43,17,-0.5,0,lw=3,color=(1,0.5,0.),head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)

#axs[0].annotate(r'$\mathrm{\textbf{S/AS}}$',xy=(0.8, 1.46), xycoords='axes fraction')
#axs[0].annotate(r'$\mathrm{\textbf{S}}$',xy=(0.87, 1.22), xycoords='axes fraction')

#D=mpimg.imread('../n7D/c3v1_c3v2_f2.png')
#imagebox1 = OffsetImage(D, zoom=0.3)
#ab1 = AnnotationBbox(imagebox1, (1.9,1.4),frameon=True,pad=0.1,xycoords='axes fraction')
#axs[0].add_artist(ab1)
##axs[0].annotate(r'$\mathrm{\textbf{A}}$',xy=(1.88, 1.14), xycoords='axes fraction')


#axs[0].annotate(r'$\mathrm{a)}$',xy=(0.0, 0.75), xycoords='figure fraction')

#axs[0].annotate(r'$\mathrm{b)}$',xy=(0.0, 0.52), xycoords='figure fraction')

plt.savefig('fig3.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()



