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

M=1000 # number of x pts
x=np.linspace(0.,5.,M)
D0=10. #Morse parameter
alpha=10. #Morse parameter
epsilon=10. #WCA parameter
sigmaA=1.   #diameter of A
sigmaAp6=sigmaA**6
sigmaAp12=sigmaAp6**2
#cutoff for wca potential
Awca=2**(1./6) *sigmaA
#minimas for Morse potential are at the WCA cutoff
Amin=Awca
#cutoffs for Morse potential
morsecutoffA=Amin+1.0*sigmaA
#assume r0 is sigma for respective pairs

FAA=np.zeros(M)
VmorseA=np.zeros(M)

urc=D0*(np.exp(-2.*alpha*(morsecutoffA-Amin))-2.*np.exp(-alpha*(morsecutoffA-Amin)))
frc=2*alpha*D0*(np.exp(-2.*alpha*(morsecutoffA-Amin))-np.exp(-alpha*(morsecutoffA-Amin)))

for i in range(M):
    if x[i]<Awca:
        FAA[i]=4.*epsilon*(sigmaAp12/x[i]**12-sigmaAp6/x[i]**6)+epsilon
    if x[i]<morsecutoffA:
        VmorseA[i]=D0*(np.exp(-2.*alpha*(x[i]-Amin))-2.*np.exp(-alpha*(x[i]-Amin)))+(x[i]-morsecutoffA)*frc-urc

fig = plt.figure(figsize=(28,7))
plt.subplots_adjust(hspace=0.0,wspace=0.27)
gs = gridspec.GridSpec(1, 3)
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2])]

#fig,ax = plt.subplots(figsize=(10,8))
axs[0].hlines(0,0,20,lw=3,zorder=1)
axs[0].plot(x,FAA,'b',label=r'$V_{\mathrm{WCA}}(r)$',lw=3,zorder=2)
axs[0].plot(x,FAA+VmorseA,'r',label=r'$V_{\mathrm{WCA}}(r)+V_{\mathrm{Morse}}(r)$',lw=3,zorder=3)

#axs[0].annotate(r'$V_{\mathrm{WCA}}$',xy=(0.10,0.7),xycoords='axes fraction',color='b')
#axs[0].annotate(r'$V_{\mathrm{WCA}}$',xy=(0.10,0.2),xycoords='axes fraction',color='r')
#axs[0].annotate(r'$+V_{\mathrm{Morse}}$',xy=(0.047,0.1),xycoords='axes fraction',color='r')
#axs[0].annotate(r'$r_{b}$',xy=(1.23,0.5),xycoords='data',color='darkorange')
#axs[0].annotate(r'$r_{c}$',xy=(morsecutoffA+0.05,0.5),xycoords='data',color='k')

axs[0].arrow(1.5,-10,0,10,lw=2,color='k',head_width=0.05,head_length=0.7,length_includes_head=True,zorder=4)
axs[0].arrow(1.5,0,0,-10,lw=2,color='k',head_width=0.05,head_length=0.7,length_includes_head=True,zorder=5)
axs[0].annotate(r'$0\leq D_{ij}$',xy=(1.55,-5),xycoords='data',color='k')
axs[0].annotate(r'$\leq 10k_{B}T$',xy=(1.55,-7),xycoords='data',color='k')

axs[0].set_ylim(-12,10)
axs[0].set_xlim(0.9,2.3)
axs[0].set_xlabel(r'$r/\sigma$')
axs[0].set_ylabel(r'$V(r)/k_{B}T$')

#legend_elements=[Line2D([0],[0],color='orange',lw=3,marker='o',markersize=15,linestyle='-',mew=2,fillstyle='none',label=r'$\langle\delta\Gamma(0)\delta\mathbf{1}(t)\rangle_{\mathbf{u}}\;t_{0}/\sigma^{2}$'),Line2D([0],[0],color='red',lw=3,marker='^',markersize=15,linestyle='-',mew=2,fillstyle='none',label=r'$\langle\delta \mathrm{M}(0)\delta\mathbf{1}(t)\rangle_{\mathbf{u}}\;t_{0}/\sigma^{2}$'),Line2D([0],[0],color="w"),Line2D([0],[0],color='k',lw=3,marker='s',markersize=15,linestyle='-',mew=2,fillstyle='none',label=r'$\langle\delta \mathrm{R}(0)\delta\mathbf{1}(t)\rangle_{\mathbf{u}}\;t_{0}/\sigma^{2}$')]
#axs[1].legend(handles=legend_elements,frameon='False',framealpha=0.0,bbox_to_anchor=(0.45, 2.55),ncol=2,columnspacing=1.0,handletextpad = 0.3,borderpad=0.0,loc='upper center')
#axs[0].legend(legend_handle, legend_labels,
#          loc = 'upper center', ncol = 3, columnspacing=1.7,handletextpad = -3.0,borderpad=0.0,frameon='False',framealpha=0.0)#,bbox_to_anchor=(0.5, 1.2))
axs[0].legend(loc = 'upper center', ncol = 1, columnspacing=1.7,handletextpad = 1.0,
        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(0.58, 1.0))

s=np.arange(0,75,5)
x=np.zeros((3,np.size(s),4))
x[0,:,:]=np.loadtxt('shearsweep/solS/run1/omega.txt')
x[1,:,:]=np.loadtxt('shearsweep/solS/run2/omega.txt')
x[2,:,:]=np.loadtxt('shearsweep/solS/run3/omega.txt')

rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]

y=np.mean(x,axis=0)
ye=np.std(x,axis=0)/3**0.5

yr=np.mean(rs,axis=0)
yre=np.std(rs,axis=0)/3**0.5

axs[1].errorbar(s,y[:,0],yerr=ye[:,0],color=(0.35,0.35,0.35),marker='o',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5)#,label=r'$\langle h_{C2v}(0)h_{Oh}(t)\rangle$')
axs[1].errorbar(s,y[:,1],yerr=ye[:,1],color=(1,0.5,0),marker='o',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5)#,label=r'$\langle h_{C2v}(0)h_{Oh}(t)\rangle$')
axs[1].errorbar(s,y[:,2],yerr=ye[:,2],color=(0.35,0.35,0.35),marker='s',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5)#,label=r'$\langle h_{C2v}(0)h_{Oh}(t)\rangle$')
axs[1].errorbar(s,y[:,3],yerr=ye[:,3],color=(1,0.5,0),marker='s',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5)#,label=r'$\langle h_{C2v}(0)h_{Oh}(t)\rangle$')
axs[1].errorbar(s,yr[:,0],yerr=yre[:,0],color=(0.35,0.35,0.35),marker='^',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5)#,label=r'$\langle h_{C2v}(0)h_{Oh}(t)\rangle$')
axs[1].errorbar(s,yr[:,1],yerr=yre[:,1],color=(1,0.5,0),marker='^',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5)#,label=r'$\langle h_{C2v}(0)h_{Oh}(t)\rangle$')
axs[1].set_yscale('log')

axs[1].set_xlabel(r'$f/f^{*}$')
axs[1].set_ylabel(r'$\langle O\rangle$')
axs[1].set_ylim(0.004,1)

x=np.zeros((3,np.size(s),4))
x[0,:,:]=np.loadtxt('shearsweep/solAS/run1/omega.txt')
x[1,:,:]=np.loadtxt('shearsweep/solAS/run2/omega.txt')
x[2,:,:]=np.loadtxt('shearsweep/solAS/run3/omega.txt')

rs=np.zeros((3,np.size(s),2))
rs[:,:,0]=x[:,:,2]/x[:,:,0]
rs[:,:,1]=x[:,:,3]/x[:,:,1]

y=np.mean(x,axis=0)
ye=np.std(x,axis=0)/3**0.5

yr=np.mean(rs,axis=0)
yre=np.std(rs,axis=0)/3**0.5

axs[2].errorbar(s,y[:,0],yerr=ye[:,0],color=(0.35,0.35,0.35),marker='o',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$Y_{\mathrm{A}}$')
axs[2].errorbar(s,yr[:,0],yerr=yre[:,0],color=(0.35,0.35,0.35),marker='^',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$k_{\mathrm{AB}}\tau$')
axs[2].errorbar(s,y[:,2],yerr=ye[:,2],color=(0.35,0.35,0.35),marker='s',linestyle='-',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$q_{\mathrm{AB}}\tau$')
axs[2].errorbar(s,y[:,1],yerr=ye[:,1],color=(1,0.5,0),marker='o',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$Y_{\mathrm{B}}$')
axs[2].errorbar(s,yr[:,1],yerr=yre[:,1],color=(1,0.5,0),marker='^',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$k_{\mathrm{BA}}\tau$')
axs[2].errorbar(s,y[:,3],yerr=ye[:,3],color=(1,0.5,0),marker='s',linestyle='--',
        lw=3,fillstyle='none',markersize=15,mew=2,capsize=5,label=r'$q_{\mathrm{BA}}\tau$')

axs[2].set_yscale('log')
axs[2].set_xlabel(r'$f/f^{*}$')
axs[2].set_ylabel(r'$\langle O\rangle$')
axs[2].set_ylim(0.004,1)

axs[2].legend(loc = 'center right', ncol = 2, columnspacing=0.5,labelspacing=2,handletextpad = 0.5,
        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(1.73, 0.6))

axs[1].annotate(r'$\mathrm{\textbf{S}}$',xy=(68,0.61),xycoords='data',color='k')
axs[2].annotate(r'$\mathrm{\textbf{AS}}$',xy=(64,0.61),xycoords='data',color='k')

x0=0.1
x,y = np.array([[x0+1,x0+1], [12,23]])
line = Line2D(x, y, lw=3, color='grey', alpha=0.7)
line.set_clip_on(False)
axs[0].add_line(line)
x,y = np.array([[x0+0.8,x0+1.2], [12,23]])
line = Line2D(x, y, lw=3, color='grey', alpha=0.7)
line.set_clip_on(False)
axs[0].add_line(line)

axs[0].arrow(x0+1,23,0.2,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
axs[0].arrow(x0+1,21.5,0.2-0.4*1.5/11,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
axs[0].arrow(x0+1,20,0.2-0.4*3/11,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
axs[0].arrow(x0+1,15,-0.2+0.4*3/11,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
axs[0].arrow(x0+1,13.5,-0.2+0.4*1.5/11,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
axs[0].arrow(x0+1,12,-0.2,0,lw=3,color='grey',alpha=0.5,head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
axs[0].annotate(r'$\mathbf{f}^{\mathrm{S}}_{i}$',xy=(0., 1.53), xycoords='axes fraction')
axs[0].annotate(r'$=fz_{i}\mathbf{\hat{x}}_{i}$',xy=(-0.1, 1.4), xycoords='axes fraction')

#adding image
#axs[0].axis('off')
soh=mpimg.imread('../editedstructures/oh_cpk_3.tga')
sc2v=mpimg.imread('../editedstructures/c2v_cpk_3.tga')
sc3v1=mpimg.imread('../editedstructures/c3v1_cpk.tga')
sc3v1=sc3v1[150:-10,40:-10,:]

imagebox1 = OffsetImage(sc2v, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (0.55,1.4),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
imagebox1 = OffsetImage(soh, zoom=0.25)
ab1 = AnnotationBbox(imagebox1, (1.3,1.4),frameon=False,xycoords='axes fraction')
axs[0].add_artist(ab1)
#adding text
axs[0].annotate(r'$\mathrm{C}_{\mathrm{2v}}(\mathrm{A})$',xy=(0.48, 1.14), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{O}_{\mathrm{h}}(\mathrm{B})$',xy=(1.23, 1.14), xycoords='axes fraction')

axs[0].arrow(1.93,19,0.5,0,lw=3,color=(0.35,0.35,0.35),head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)
axs[0].arrow(2.43,17,-0.5,0,lw=3,color=(1,0.5,0.),head_width=0.7,head_length=0.06,length_includes_head=True,clip_on=False)

axs[0].annotate(r'$\mathrm{\textbf{S/AS}}$',xy=(0.8, 1.46), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{\textbf{S}}$',xy=(0.87, 1.22), xycoords='axes fraction')

Doct=mpimg.imread('../n6D/oct.png')
Dpolytd=mpimg.imread('../n6D/polytd2.png')
DsolC=mpimg.imread('../n6D/solC2.png')
DsolB=mpimg.imread('../n6D/solB2.png')

imagebox1 = OffsetImage(Dpolytd, zoom=0.3)
ab1 = AnnotationBbox(imagebox1, (1.9,1.4),frameon=True,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].annotate(r'$\mathrm{\textbf{A}}$',xy=(1.88, 1.14), xycoords='axes fraction')

imagebox1 = OffsetImage(Doct, zoom=0.3)
ab1 = AnnotationBbox(imagebox1, (2.4,1.4),frameon=True,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].annotate(r'$\mathrm{\textbf{B}}$',xy=(2.38, 1.14), xycoords='axes fraction')

imagebox1 = OffsetImage(DsolC, zoom=0.3)
ab1 = AnnotationBbox(imagebox1, (2.9,1.4),frameon=True,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].annotate(r'$\mathrm{\textbf{S}}$',xy=(2.88, 1.14), xycoords='axes fraction')

imagebox1 = OffsetImage(DsolB, zoom=0.3)
ab1 = AnnotationBbox(imagebox1, (3.4,1.4),frameon=True,pad=0.1,xycoords='axes fraction')
axs[0].add_artist(ab1)
axs[0].annotate(r'$\mathrm{\textbf{AS}}$',xy=(3.35, 1.14), xycoords='axes fraction')

cpick = cm.ScalarMappable(cmap='Blues')
cpick.set_array([])
axs[0].imshow([[1,1], [0,0]],cmap = 'Blues',interpolation = 'bicubic',
        extent=[6.15,6.25,11.5,22.5],aspect='auto',clip_on=False)

axs[0].annotate(r'$\mathrm{0k_{B}T}$',xy=(3.85, 1.07), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{5k_{B}T}$',xy=(3.85, 1.295), xycoords='axes fraction')
axs[0].annotate(r'$\mathrm{10k_{B}T}$',xy=(3.85, 1.52), xycoords='axes fraction')

#axs[0].annotate(r'$\mathrm{a)}$',xy=(0.0, 0.75), xycoords='figure fraction')

#axs[0].annotate(r'$\mathrm{b)}$',xy=(0.0, 0.52), xycoords='figure fraction')

plt.savefig('fig1.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()

#np.savetxt('fig1_wca.txt',np.stack((x,FAA),axis=1))
#np.savetxt('fig1_wca_morse.txt',np.stack((x,FAA+VmorseA),axis=1))


