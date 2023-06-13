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

fig = plt.figure(figsize=(28,7))
plt.subplots_adjust(hspace=0.0,wspace=0.27)
gs = gridspec.GridSpec(1, 3)
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2])]

#axs[0].legend(loc = 'upper center', ncol = 1, columnspacing=1.7,handletextpad = 1.0,
#        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(0.58, 1.0))

dx=0.13
x0=0.3
x1=0.7
y0=-0.25

m=mpimg.imread('alphmatch/c2_c2v_f2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0,y0+0.75),frameon=True,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].annotate("", xy=(x0-dx, y0+0.95), xytext=(x0+dx, y0+0.95),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))
axs[0].annotate("", xy=(x0-dx, y0+0.92), xytext=(x0+dx, y0+0.92),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='--'))

m=mpimg.imread('alphmatch/c3v1_c2_f_symm2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x1,y0+0.75),frameon=True,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].annotate("", xy=(x1-dx, y0+0.95), xytext=(x1+dx, y0+0.95),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))
axs[0].annotate("", xy=(x1-dx, y0+0.92), xytext=(x1+dx, y0+0.92),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='--'))

m=mpimg.imread('alphmatch/c3v1_c2v_f_st2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0,y0+0.35),frameon=True,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].annotate("", xy=(x0-dx, y0+0.55), xytext=(x0+dx, y0+0.55),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))
axs[0].annotate("", xy=(x0-dx, y0+0.52), xytext=(x0+dx, y0+0.52),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='--'))


m=mpimg.imread('alphmatch/c3v1_c3v2_f2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x1,y0+0.35),frameon=True,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].annotate("", xy=(x1-dx, y0+0.55), xytext=(x1+dx, y0+0.55),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='brown',linestyle='-'))
axs[0].annotate("", xy=(x1-dx, y0+0.52), xytext=(x1+dx, y0+0.52),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='brown',linestyle='--'))

m=mpimg.imread('alphmatch/c3v2_c2_f2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (0.5*x0+0.5*x1,y0-0.05),frameon=True,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].annotate("", xy=(0.5*x0+0.5*x1-dx, y0+0.15), xytext=(0.5*x0+0.5*x1+dx, y0+0.15),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))
axs[0].annotate("", xy=(0.5*x0+0.5*x1-dx, y0+0.12), xytext=(0.5*x0+0.5*x1+dx, y0+0.12),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='--'))

#second column
m=mpimg.imread('alphmatch/c2_c3v1_f2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0,y0+0.75),frameon=True,pad=0.1,xycoords='axes fraction')
axs[1].add_artist(ab1)
axs[1].annotate("", xy=(x0-dx, y0+0.935), xytext=(x0+dx, y0+0.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))

m=mpimg.imread('alphmatch/c2_c3v2_f2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x1,y0+0.75),frameon=True,pad=0.1,xycoords='axes fraction')
axs[1].add_artist(ab1)
axs[1].annotate("", xy=(x1-dx, y0+0.935), xytext=(x1+dx, y0+0.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))

m=mpimg.imread('alphmatch/c2v_c2_f2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0,y0+0.35),frameon=True,pad=0.1,xycoords='axes fraction')
axs[1].add_artist(ab1)
axs[1].annotate("", xy=(x0-dx, y0+0.535), xytext=(x0+dx, y0+0.535),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))

m=mpimg.imread('alphmatch/c2v_c3v2_f2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x1,y0+0.35),frameon=True,pad=0.1,xycoords='axes fraction')
axs[1].add_artist(ab1)
axs[1].annotate("", xy=(x1-dx, y0+0.535), xytext=(x1+dx, y0+0.535),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='k',linestyle='-'))

m=mpimg.imread('alphmatch/c3v1_c2v_f2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0,y0-0.05),frameon=True,pad=0.1,xycoords='axes fraction')
axs[1].add_artist(ab1)
axs[1].annotate("", xy=(x0-dx, y0+0.135), xytext=(x0+dx, y0+0.135),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))

m=mpimg.imread('alphmatch/c2v_c3v2_f2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x1,y0-0.05),frameon=True,pad=0.1,xycoords='axes fraction')
axs[1].add_artist(ab1)
axs[1].annotate("", xy=(x1-dx, y0+0.135), xytext=(x1+dx, y0+0.135),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='dodgerblue',linestyle='-'))

m=mpimg.imread('alphmatch/c2_c2v_f_asymm2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (0.5*x0+0.5*x1,y0-0.45),frameon=True,pad=0.1,xycoords='axes fraction')
axs[1].add_artist(ab1)
axs[1].annotate("", xy=(0.5*x0+0.5*x1-dx, y0-0.265), xytext=(0.5*x0+0.5*x1+dx, y0-0.265),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='orange',linestyle='-'))

#third column
m=mpimg.imread('alphmatch/c2_c2v_c2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0,y0+0.75),frameon=True,pad=0.1,xycoords='axes fraction')
axs[2].add_artist(ab1)
axs[2].annotate("", xy=(x0-dx, y0+0.935), xytext=(x0+dx, y0+0.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='orange',linestyle='-'))
axs[2].annotate("", xy=(x0-dx, y0+0.935), xytext=(x0-dx+0.04, y0+0.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='orange',linestyle='-'))

m=mpimg.imread('alphmatch/c2_c3v1_c2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x1,y0+0.75),frameon=True,pad=0.1,xycoords='axes fraction')
axs[2].add_artist(ab1)
axs[2].annotate("", xy=(x1-dx, y0+0.935), xytext=(x1+dx, y0+0.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))
axs[2].annotate("", xy=(x1-dx, y0+0.935), xytext=(x1-dx+0.04, y0+0.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))

m=mpimg.imread('alphmatch/c2_c3v2_c2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0,y0+0.35),frameon=True,pad=0.1,xycoords='axes fraction')
axs[2].add_artist(ab1)
axs[2].annotate("", xy=(x0-dx, y0+0.535), xytext=(x0+dx, y0+0.535),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))
axs[2].annotate("", xy=(x0-dx, y0+0.535), xytext=(x0-dx+0.04, y0+0.535),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))

m=mpimg.imread('alphmatch/c2v_c2_c2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x1,y0+0.35),frameon=True,pad=0.1,xycoords='axes fraction')
axs[2].add_artist(ab1)
axs[2].annotate("", xy=(x1-dx, y0+0.535), xytext=(x1+dx, y0+0.535),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))
axs[2].annotate("", xy=(x1-dx, y0+0.535), xytext=(x1-dx+0.04, y0+0.535),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))

m=mpimg.imread('alphmatch/c3v1_c2v_c2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x0,y0-0.05),frameon=True,pad=0.1,xycoords='axes fraction')
axs[2].add_artist(ab1)
axs[2].annotate("", xy=(x0-dx, y0+0.135), xytext=(x0+dx, y0+0.135),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))
axs[2].annotate("", xy=(x0-dx, y0+0.135), xytext=(x0-dx+0.04, y0+0.135),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))

m=mpimg.imread('alphmatch/c3v2_c2v_c2.png')
imagebox1 = OffsetImage(m, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (x1,y0-0.05),frameon=True,pad=0.1,xycoords='axes fraction')
axs[2].add_artist(ab1)
axs[2].annotate("", xy=(x1-dx, y0+0.135), xytext=(x1+dx, y0+0.135),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='dodgerblue',linestyle='-'))
axs[2].annotate("", xy=(x1-dx, y0+0.135), xytext=(x1-dx+0.04, y0+0.135),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='dodgerblue',linestyle='-'))

#axs[0].plot(np.arange(10),np.arange(10))
axs[1].set_xlim(0,1)
axs[1].set_ylim(0,1)

cpick = cm.ScalarMappable(cmap='Blues')
cpick.set_array([])
axs[1].imshow([[0,1], [0,1]],cmap = 'Blues',interpolation = 'bicubic',
        extent=[-0.25,1.25,0.9,1.],aspect='auto',clip_on=False)

axs[1].annotate(r'$\mathrm{0k_{B}T}$',xy=(-0.3, 0.8), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{5k_{B}T}$',xy=(0.45, 0.8), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{10k_{B}T}$',xy=(1.2, 0.8), xycoords='axes fraction')

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

imagebox1 = OffsetImage(sc3v1, zoom=0.27)
ab1 = AnnotationBbox(imagebox1, (0.1,1.7),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc3v2, zoom=0.25,zorder=2)
ab1 = AnnotationBbox(imagebox1, (0.8,1.6),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (0.45,1.9),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2v, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (0.45,1.3),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
ab1.zorder=1

#adding text
axs[0].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(-0.13, 1.6), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(0.9, 1.6), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{C}_{\mathrm{2}}$',xy=(0.43, 2.2), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{C}_{\mathrm{2v}}$',xy=(0.43, 1.1), xycoords='axes fraction')

axs[0].annotate("", xy=(0.38, 1.92), xytext=(0.20, 1.71),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))
axs[0].annotate("", xy=(0.36, 1.96), xytext=(0.18, 1.75),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='--'))
axs[0].annotate("", xy=(0.25, 1.65), xytext=(0.68, 1.65),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='brown',linestyle='-'))
axs[0].annotate("", xy=(0.25, 1.6), xytext=(0.68, 1.6),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='brown',linestyle='--'))
axs[0].annotate("", xy=(0.45, 1.45), xytext=(0.45, 1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='--'))
axs[0].annotate("", xy=(0.49, 1.45), xytext=(0.49, 1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))
axs[0].annotate("", xy=(0.15, 1.49), xytext=(0.32, 1.3),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='--'))
axs[0].annotate("", xy=(0.18, 1.51), xytext=(0.35, 1.32),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))
axs[0].annotate("", xy=(0.72, 1.71), xytext=(0.54, 1.91),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))
axs[0].annotate("", xy=(0.73, 1.76), xytext=(0.55, 1.96),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='--'))

imagebox1 = OffsetImage(sc3v1, zoom=0.27)
ab1 = AnnotationBbox(imagebox1, (0.1,1.7),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc3v2, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (0.8,1.6),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (0.45,1.9),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2v, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (0.45,1.3),frameon=False,xycoords='axes fraction')
axs[1].add_artist(ab1)
ab1.zorder=1

#adding text
axs[1].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(-0.13, 1.6), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(0.9, 1.6), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{C}_{\mathrm{2}}$',xy=(0.43, 2.2), xycoords='axes fraction')
axs[1].annotate(r'$\mathrm{C}_{\mathrm{2v}}$',xy=(0.43, 1.1), xycoords='axes fraction')

axs[1].annotate("", xy=(0.37, 1.94), xytext=(0.19, 1.73),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))
axs[1].annotate("", xy=(0.44, 1.45), xytext=(0.44, 1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='orange',linestyle='-'))
axs[1].annotate("", xy=(0.49, 1.45), xytext=(0.49, 1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))
axs[1].annotate("", xy=(0.165, 1.5), xytext=(0.335, 1.31),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))
axs[1].annotate("", xy=(0.725, 1.735), xytext=(0.545, 1.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))
axs[1].annotate("", xy=(0.72, 1.51), xytext=(0.555, 1.34),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='dodgerblue',linestyle='-'))
axs[1].annotate("", xy=(0.76, 1.49), xytext=(0.585, 1.31),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='k',linestyle='-'))

imagebox1 = OffsetImage(sc3v1, zoom=0.27)
ab1 = AnnotationBbox(imagebox1, (0.1,1.7),frameon=False,xycoords='axes fraction')
axs[2].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc3v2, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (0.8,1.6),frameon=False,xycoords='axes fraction')
axs[2].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (0.45,1.9),frameon=False,xycoords='axes fraction')
axs[2].add_artist(ab1)
ab1.zorder=1
imagebox1 = OffsetImage(sc2v, zoom=0.45)
ab1 = AnnotationBbox(imagebox1, (0.45,1.3),frameon=False,xycoords='axes fraction')
axs[2].add_artist(ab1)
ab1.zorder=1

#adding text
axs[2].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(-0.13, 1.6), xycoords='axes fraction')
axs[2].annotate(r'$\mathrm{C}_{\mathrm{3v}}$',xy=(0.9, 1.6), xycoords='axes fraction')
axs[2].annotate(r'$\mathrm{C}_{\mathrm{2}}$',xy=(0.43, 2.2), xycoords='axes fraction')
axs[2].annotate(r'$\mathrm{C}_{\mathrm{2v}}$',xy=(0.43, 1.1), xycoords='axes fraction')

axs[2].annotate("", xy=(0.37, 1.94), xytext=(0.19, 1.73),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))
axs[2].annotate("", xy=(0.37, 1.94), xytext=(0.37+(0.19-0.37)/10, 1.94+(1.73-1.94)/10),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='blueviolet',linestyle='-'))

axs[2].annotate("", xy=(0.44, 1.45), xytext=(0.44, 1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='orange',linestyle='-'))
axs[2].annotate("", xy=(0.44, 1.85+(1.45-1.85)/10), xytext=(0.44, 1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='orange',linestyle='-'))

axs[2].annotate("", xy=(0.49, 1.45), xytext=(0.49, 1.85),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))
axs[2].annotate("", xy=(0.49, 1.45), xytext=(0.49, 1.45+(1.85-1.45)/10),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='green',linestyle='-'))

axs[2].annotate("", xy=(0.165, 1.5), xytext=(0.335, 1.31),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))
axs[2].annotate("", xy=(0.165, 1.5), xytext=(0.165+(0.335-0.165)/10, 1.5+(1.31-1.5)/10),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='deeppink',linestyle='-'))

axs[2].annotate("", xy=(0.725, 1.735), xytext=(0.545, 1.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))
axs[2].annotate("", xy=(0.545+(0.725-0.545)/10, 1.935+(1.735-1.935)/10), xytext=(0.545, 1.935),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34",lw=3,color='lime',linestyle='-'))

axs[2].annotate("", xy=(0.74, 1.5), xytext=(0.57, 1.325),clip_on=False,zorder=2,xycoords='axes fraction',
        arrowprops=dict(arrowstyle="<|-,head_width=0.24,head_length=0.34",lw=3,color='dodgerblue',linestyle='-'))
axs[2].annotate("", xy=(0.74, 1.5), xytext=(0.74+(0.57-0.74)/10, 1.5+(1.325-1.5)/10),clip_on=False,zorder=2,xycoords='axes fraction',
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

axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')

plt.savefig('sifig4.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()



