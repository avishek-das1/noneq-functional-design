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
from numba import jit

#mpl.style.use("classic")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams.update({'font.size': 32})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,bm}')

fig = plt.figure(figsize=(28,14))
#plt.subplots_adjust(hspace=0.25,wspace=30)
gs = gridspec.GridSpec(2, 24,hspace=0.25,wspace=30,left=0.14)
gsb = gridspec.GridSpec(2, 24,hspace=0.25,wspace=30)
axs=[fig.add_subplot(gs[0,0:6]),fig.add_subplot(gs[0,6:12]),fig.add_subplot(gs[0,15:21]),
        fig.add_subplot(gs[1,0:6]),fig.add_subplot(gs[1,6:12]),fig.add_subplot(gs[1,15:21])]
axsb=[fig.add_subplot(gsb[0,0:6]),fig.add_subplot(gsb[0,6:12]),fig.add_subplot(gsb[0,15:21]),
        fig.add_subplot(gsb[1,0:6]),fig.add_subplot(gsb[1,6:12]),fig.add_subplot(gsb[1,15:21])]
axins = inset_axes(axsb[0], width='60%',height='40%',bbox_to_anchor=(280,380,550,640))

axs[0].axis('off')
axs[3].axis('off')
axsb[1].axis('off')
axsb[2].axis('off')
axsb[4].axis('off')
axsb[5].axis('off')

#axs[2].legend(loc = 'center right', ncol = 2, columnspacing=0.5,labelspacing=2,handletextpad = 0.5,
#        borderpad=0.0,frameon='False',framealpha=0.0,bbox_to_anchor=(1.73, 0.6))

#adding image
mp=mpimg.imread('microphase7.tga')
mp=mp[380:-200,385:-240,:]

axins.imshow(mp)
axins.xaxis.set_ticks([])
axins.yaxis.set_ticks([])

mp2=mpimg.imread('single.tga')
mp2=mp2[190:-70,10:-7,:]

imagebox1 = OffsetImage(mp2, zoom=0.25,clip_on=False)
ab1 = AnnotationBbox(imagebox1, (0.6,0.83),frameon=False,xycoords='axes fraction')
axins.add_artist(ab1)

a1 = mpl.patches.FancyArrowPatch((260, 140), (370, 70),
        connectionstyle="arc3,rad=-.3",color='orange',
        arrowstyle="fancy,head_width=20,head_length=20,tail_width=10",
        lw=2,clip_on=False,zorder=11)
axins.add_patch(a1)

a2 = mpl.patches.FancyArrowPatch((450, 70), (530, 130),
        connectionstyle="arc3,rad=-.3",color='orange',
        arrowstyle="fancy,head_width=20,head_length=20,tail_width=10",
        lw=2,clip_on=False,zorder=11)
axins.add_patch(a2)

#axs[0].axis('off')
#imagebox1 = OffsetImage(mp, zoom=0.45,clip_on=False)
#ab1 = AnnotationBbox(imagebox1, (0.5,0.5),frameon=True,pad=0.04,xycoords='axes fraction')
#axs[0].add_artist(ab1)


M=1000 # number of x pts
x=np.linspace(0.,5.,M)
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
Ycutoff=5.0*sigmaA

D0=10.0 #Morse parameter
Ye=10.0
Yk=0.1
VAA=np.zeros(M)
VmorseA=np.zeros(M)
VY=np.zeros(M)
urc=D0*(np.exp(-2.*alpha*(morsecutoffA-Amin))-2.*np.exp(-alpha*(morsecutoffA-Amin)))
frc=2*alpha*D0*(np.exp(-2.*alpha*(morsecutoffA-Amin))-np.exp(-alpha*(morsecutoffA-Amin)))
urcY=Ye*np.exp(-Yk*Ycutoff)/Ycutoff
frcY=Ye*np.exp(-Yk*Ycutoff)/Ycutoff**2 *(1+Yk*Ycutoff)
for i in range(M):
    if x[i]<Awca:
        VAA[i]=4.*epsilon*(sigmaAp12/x[i]**12-sigmaAp6/x[i]**6)+epsilon
    if x[i]<morsecutoffA:
        VmorseA[i]=D0*(np.exp(-2.*alpha*(x[i]-Amin))-2.*np.exp(-alpha*(x[i]-Amin)))+(x[i]-morsecutoffA)*frc-urc
    if x[i]<Ycutoff:
        VY[i]=Ye*np.exp(-Yk*x[i])/x[i]++(x[i]-Ycutoff)*frcY-urcY

axsb[0].hlines(0,0,5,'k',alpha=0.2)
axsb[0].hlines(-4.6,1.122,1.4,'k',linestyle='--')
axsb[0].hlines(2.8,1.4,2.5,'k',linestyle='--')
axsb[0].plot(x,VAA+VmorseA+VY,'r',label=r'$V_{\mathrm{WCA+M+Y}}(r)$')
axsb[0].set_xlim(1.0,3.0)
axsb[0].set_ylim(-5,5)
axsb[0].arrow(1.4,2.8,0,-7.4,lw=1,color='k',head_width=0.05,head_length=0.5,length_includes_head=True,zorder=4)
axsb[0].arrow(1.4,-4.6,0,7.4,lw=1,color='k',head_width=0.05,head_length=0.5,length_includes_head=True,zorder=4)
axsb[0].arrow(2.5,2.8,0,-2.8,lw=1,color='k',head_width=0.05,head_length=0.5,length_includes_head=True,zorder=4)
axsb[0].arrow(2.5,0,0,2.8,lw=1,color='k',head_width=0.05,head_length=0.5,length_includes_head=True,zorder=4)

axsb[0].set_xlabel(r'$r/\sigma$')
axsb[0].set_ylabel(r'$V(r)/k_{\mathrm{B}}T$')
axsb[0].annotate(r'$\Delta E_{\mathrm{d}}$',xy=(1.45,1.3),xycoords='data',color='k')
axsb[0].annotate(r'$\Delta E_{\mathrm{a}}$',xy=(2.55,1.3),xycoords='data',color='k')

y=np.loadtxt('catalysis/new5/Fsh0/labels.txt')
u=np.loadtxt('catalysis/new6/Fsh25/labels.txt')
axsb[3].plot(np.arange(10000)*1000*5e-5,y/300,color='brown',lw=3,label='$f/f^{*}=0$')
axsb[3].plot(np.arange(5000)*1000*5e-5,u/300,color='g',lw=3,label='$f/f^{*}=25$')

y=np.loadtxt('catalysis/new5/Fsh0b/labels.txt')
u=np.loadtxt('catalysis/new6/Fsh25b/labels.txt')
axsb[3].plot(np.arange(10000)*1000*5e-5,y/300,color='brown',lw=3)
axsb[3].plot(np.arange(5000)*1000*5e-5,u/300,color='g',lw=3)

y=np.loadtxt('catalysis/new5/Fsh0c/labels.txt')
u=np.loadtxt('catalysis/new6/Fsh25c/labels.txt')
axsb[3].plot(np.arange(10000)*1000*5e-5,y/300,color='brown',lw=3)
axsb[3].plot(np.arange(5000)*1000*5e-5,u/300,color='g',lw=3)


axsb[3].set_xlabel(r'$t/t^{*}$')
axsb[3].set_ylabel(r'$Y_{c}$')
axsb[3].legend(loc = 'center right', ncol = 1, columnspacing=0.5,labelspacing=0.5,handletextpad = 0.5,
        borderpad=0.0,frameon='False',framealpha=0.0)#,bbox_to_anchor=(0.73, 0.6))
axsb[3].set_xlim(0,250)
axsb[3].set_ylim(0,1)

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

@jit
def gausscg(cmax,cb,ngridx,ngridy,xi,yi,xb):
    den=np.zeros((ngridy,ngridx))
    fn=np.zeros((ngridy,ngridx,3))
    vg=0.15
    for ii in range(ngridx):
        for jj in range(ngridy):
            for i in range(cmax):
                den[jj,ii]+=np.exp((-(cb[i,0]-xi[ii])**2-(cb[i,1]-yi[jj])**2)/2/vg)/(2.*np.pi*vg)**0.5
                fn[jj,ii,:]+=xb[i,:]*np.exp((-(cb[i,0]-xi[ii])**2-(cb[i,1]-yi[jj])**2)/2/vg)/(2.*np.pi*vg)**0.5
            fn[jj,ii,:]/=den[jj,ii]
    return fn,den

lta=340
ltb=90
lt=lta+ltb
M=24
N=90
x=np.zeros((lt,3))
xb=np.zeros((lt,3))
ct=np.zeros((lt))
Ds=np.zeros((lt))
Yeks=np.zeros((lt,2))
for i in range(8):
    y=np.loadtxt('paramsweep/shear1_'+str(i+1)+'/omega.txt')
    D=np.loadtxt('paramsweep/shear1_'+str(i+1)+'/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear1_'+str(i+1)+'/Yekrunning.txt')
    l=int(np.size(y)/3)
    x[:l,:]+=y
    Ds[:l]=D
    Yeks[:l,:]=Yek
    ct[:l]+=1
for i in range(8,12):
    y=np.loadtxt('paramsweep/shear1_'+str(i+1-8)+'/run2/omega.txt')
    D=np.loadtxt('paramsweep/shear1_'+str(i+1-8)+'/run2/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear1_'+str(i+1-8)+'/run2/Yekrunning.txt')
    l=int(np.size(y)/3)
    x[:l,:]+=y
    Ds[:l]=D
    Yeks[:l,:]=Yek
    ct[:l]+=1
for i in range(12,16):
    y=np.loadtxt('paramsweep/shear1_'+str(i+1-12)+'/run3/omega.txt')
    D=np.loadtxt('paramsweep/shear1_'+str(i+1-12)+'/run3/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear1_'+str(i+1-12)+'/run3/Yekrunning.txt')
    l=int(np.size(y)/3)
    x[:l,:]+=y
    Ds[:l]=D
    Yeks[:l,:]=Yek
    ct[:l]+=1
for i in range(16,20):
    y=np.loadtxt('paramsweep/shear1_'+str(i+1-16)+'/run4/omega.txt')
    D=np.loadtxt('paramsweep/shear1_'+str(i+1-16)+'/run4/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear1_'+str(i+1-16)+'/run4/Yekrunning.txt')
    l=int(np.size(y)/3)
    fl=int(np.floor(l/ltb))
    left=l-fl*ltb
    for i in range(fl):
        x[lta:,:]+=y[i*ltb:(i+1)*ltb,:]
        Ds[lta:]=D[i*ltb:(i+1)*ltb]
        Yeks[lta:,:]=Yek[i*ltb:(i+1)*ltb,:]
        ct[lta:]+=1
    x[lta:lta+left,:]+=y[fl*ltb:,:]
    Ds[lta:lta+left]=D[fl*ltb:]
    Yeks[lta:lta+left,:]=Yek[fl*ltb:,:]
    ct[lta:lta+left]+=1
for i in range(20,M):
    y=np.loadtxt('paramsweep/shear1_'+str(i+1-20)+'/run5/omega.txt')
    D=np.loadtxt('paramsweep/shear1_'+str(i+1-20)+'/run5/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear1_'+str(i+1-20)+'/run5/Yekrunning.txt')
    l=int(np.size(y)/3)
    fl=int(np.floor(l/ltb))
    left=l-fl*ltb
    for i in range(fl):
        x[lta:,:]+=y[i*ltb:(i+1)*ltb,:]
        Ds[lta:]=D[i*ltb:(i+1)*ltb]
        Yeks[lta:,:]=Yek[i*ltb:(i+1)*ltb,:]
        ct[lta:]+=1
    x[lta:lta+left,:]+=y[fl*ltb:,:]
    Ds[lta:lta+left]=D[fl*ltb:]
    Yeks[lta:lta+left,:]=Yek[fl*ltb:,:]
    ct[lta:lta+left]+=1

xb[:,0]=x[:,0]/ct   
xb[:,1]=x[:,1]/ct
xb[:,2]=x[:,2]/ct

l=lt
cb=np.zeros((l,2))
for i in range(l):
    cb[i,0],cb[i,1]=hillbarrier(Ds[i],Yeks[i,0],Yeks[i,1])

ngridx = 100
ngridy = 100

xl=0
xh=3.2
yl=2
yh=7.5

xi = np.linspace(xl, xh, ngridx)
yi = np.linspace(yl, yh, ngridy)

fn,den=gausscg(lt,cb,ngridx,ngridy,xi,yi,xb)

#xx,yy=np.meshgrid(xi,yi)
im1=axs[1].contourf(xi,yi,fn[:,:,0]/N,levels=10,vmin=0,cmap='viridis')
#plt.plot(ch,bh,color='r',lw=1)
#axs[1].plot(cb[:,0], cb[:,1], marker='o',c='orange',linestyle='None',ms=3)
#plt.colorbar()
axs[1].set_xlabel(r'$\Delta E_{\mathrm{a}}/k_{\mathrm{B}}T$')
axs[1].set_ylabel(r'$\Delta E_{\mathrm{d}}/k_{\mathrm{B}}T$')
axs[1].set_xlim(xl, xh)
axs[1].set_ylim(yl, yh)

im4=axs[4].contourf(xi,yi,fn[:,:,2]/N,levels=10,vmin=0,cmap='viridis')
#plt.plot(ch,bh,color='r',lw=1)
#axs[1].plot(cb[:,0], cb[:,1], marker='o',c='orange',linestyle='None',ms=3)
#plt.colorbar()
axs[4].set_xlabel(r'$\Delta E_{\mathrm{a}}/k_{\mathrm{B}}T$')
axs[4].set_ylabel(r'$\Delta E_{\mathrm{d}}/k_{\mathrm{B}}T$')
axs[4].set_xlim(xl, xh)
axs[4].set_ylim(yl, yh)

x=np.zeros((lt,3))
xb=np.zeros((lt,3))
ct=np.zeros((lt))
Ds=np.zeros((lt))
Yeks=np.zeros((lt,2))
for i in range(8):
    y=np.loadtxt('paramsweep/shear25_'+str(i+1)+'/omega.txt')
    D=np.loadtxt('paramsweep/shear25_'+str(i+1)+'/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear25_'+str(i+1)+'/Yekrunning.txt')
    l=int(np.size(y)/3)
    x[:l,:]+=y
    Ds[:l]=D
    Yeks[:l,:]=Yek
    ct[:l]+=1
for i in range(8,12):
    y=np.loadtxt('paramsweep/shear25_'+str(i+1-8)+'/run2/omega.txt')
    D=np.loadtxt('paramsweep/shear25_'+str(i+1-8)+'/run2/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear25_'+str(i+1-8)+'/run2/Yekrunning.txt')
    l=int(np.size(y)/3)
    x[:l,:]+=y
    Ds[:l]=D
    Yeks[:l,:]=Yek
    ct[:l]+=1
for i in range(12,16):
    y=np.loadtxt('paramsweep/shear25_'+str(i+1-12)+'/run3/omega.txt')
    D=np.loadtxt('paramsweep/shear25_'+str(i+1-12)+'/run3/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear25_'+str(i+1-12)+'/run3/Yekrunning.txt')
    l=int(np.size(y)/3)
    x[:l,:]+=y
    Ds[:l]=D
    Yeks[:l,:]=Yek
    ct[:l]+=1
for i in range(16,20):
    y=np.loadtxt('paramsweep/shear25_'+str(i+1-16)+'/run4/omega.txt')
    D=np.loadtxt('paramsweep/shear25_'+str(i+1-16)+'/run4/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear25_'+str(i+1-16)+'/run4/Yekrunning.txt')
    l=int(np.size(y)/3)
    fl=int(np.floor(l/ltb))
    left=l-fl*ltb
    for i in range(fl):
        x[lta:,:]+=y[i*ltb:(i+1)*ltb,:]
        Ds[lta:]=D[i*ltb:(i+1)*ltb]
        Yeks[lta:,:]=Yek[i*ltb:(i+1)*ltb,:]
        ct[lta:]+=1
    x[lta:lta+left,:]+=y[fl*ltb:,:]
    Ds[lta:lta+left]=D[fl*ltb:]
    Yeks[lta:lta+left,:]=Yek[fl*ltb:,:]
    ct[lta:lta+left]+=1
for i in range(20,M):
    y=np.loadtxt('paramsweep/shear25_'+str(i+1-20)+'/run5/omega.txt')
    D=np.loadtxt('paramsweep/shear25_'+str(i+1-20)+'/run5/D0running.txt')
    Yek=np.loadtxt('paramsweep/shear25_'+str(i+1-20)+'/run5/Yekrunning.txt')
    l=int(np.size(y)/3)
    fl=int(np.floor(l/ltb))
    left=l-fl*ltb
    for i in range(fl):
        x[lta:,:]+=y[i*ltb:(i+1)*ltb,:]
        Ds[lta:]=D[i*ltb:(i+1)*ltb]
        Yeks[lta:,:]=Yek[i*ltb:(i+1)*ltb,:]
        ct[lta:]+=1
    x[lta:lta+left,:]+=y[fl*ltb:,:]
    Ds[lta:lta+left]=D[fl*ltb:]
    Yeks[lta:lta+left,:]=Yek[fl*ltb:,:]
    ct[lta:lta+left]+=1

xb[:,0]=x[:,0]/ct   
xb[:,1]=x[:,1]/ct
xb[:,2]=x[:,2]/ct

l=lt
cb=np.zeros((l,2))
for i in range(l):
    cb[i,0],cb[i,1]=hillbarrier(Ds[i],Yeks[i,0],Yeks[i,1])

ngridx = 100
ngridy = 100

xl=0
xh=3.2
yl=2
yh=7.5

xi = np.linspace(xl, xh, ngridx)
yi = np.linspace(yl, yh, ngridy)

fn,den=gausscg(lt,cb,ngridx,ngridy,xi,yi,xb)

#xx,yy=np.meshgrid(xi,yi)
im2=axs[2].contourf(xi,yi,fn[:,:,0]/N,levels=10,vmin=0,cmap='viridis')
#plt.plot(ch,bh,color='r',lw=1)
#axs[1].plot(cb[:,0], cb[:,1], marker='o',c='orange',linestyle='None',ms=3)
#plt.colorbar()
axs[2].set_xlabel(r'$\Delta E_{\mathrm{a}}/k_{\mathrm{B}}T$')
axs[2].set_ylabel(r'$\Delta E_{\mathrm{d}}/k_{\mathrm{B}}T$')
axs[2].set_xlim(xl, xh)
axs[2].set_ylim(yl, yh)

im5=axs[5].contourf(xi,yi,fn[:,:,2]/N,levels=10,vmin=0,cmap='viridis')
#plt.plot(ch,bh,color='r',lw=1)
#axs[1].plot(cb[:,0], cb[:,1], marker='o',c='orange',linestyle='None',ms=3)
#plt.colorbar()
axs[5].set_xlabel(r'$\Delta E_{\mathrm{a}}/k_{\mathrm{B}}T$')
axs[5].set_ylabel(r'$\Delta E_{\mathrm{d}}/k_{\mathrm{B}}T$')
axs[5].set_xlim(xl, xh)
axs[5].set_ylim(yl, yh)

cax=fig.add_axes([0.515,0.539,0.017,0.343])
plt.colorbar(im1,cax)

cax=fig.add_axes([0.812,0.539,0.017,0.343])
plt.colorbar(im2,cax)

cax=fig.add_axes([0.515,0.112,0.017,0.343])
cbar=plt.colorbar(im4,cax,ticks=[0,1e-8,2e-8,3e-8,4e-8])
cbar.ax.set_yticklabels([r'$0\times10^{-8}$', r'$1\times10^{-8}$',r'$2\times10^{-8}$', r'$3\times10^{-8}$',r'$4\times 10^{-8}$'])
#cax.get_xaxis().get_offset_text().set_position((1,1))

cax=fig.add_axes([0.812,0.112,0.017,0.343])
cbar=plt.colorbar(im5,cax,ticks=[0,1e-7,2e-7,3e-7])
cbar.ax.set_yticklabels([r'$0\times10^{-7}$', r'$1\times10^{-7}$',r'$2\times10^{-7}$', r'$3\times10^{-7}$'])
#plt.colorbar(im1,bbox_to_anchor=(1.73, 0.6),clip_on=False)
#axs[1].colorbar(loc = 'center right', bbox_to_anchor=(1.73, 0.6),clip_on=False,)

axs[1].annotate(r'$Y_{6}$',xy=(1.1,1.05),xycoords='axes fraction',clip_on=False,color='k')
axs[2].annotate(r'$Y_{6}$',xy=(1.1,1.05),xycoords='axes fraction',clip_on=False,color='k')
axs[1].annotate(r'$q_{6}\tau$',xy=(1.1,-0.2),xycoords='axes fraction',clip_on=False,color='k')
axs[2].annotate(r'$q_{6}\tau$',xy=(1.1,-0.2),xycoords='axes fraction',clip_on=False,color='k')

axs[1].annotate(r'$\bm{f/f^{*}=1}$',xy=(1.6,2.7),xycoords='data',color='white')
axs[2].annotate(r'$\bm{f/f^{*}=25}$',xy=(1.6,2.7),xycoords='data',color='white')
axs[4].annotate(r'$\bm{f/f^{*}=1}$',xy=(1.6,2.7),xycoords='data',color='white')
axs[5].annotate(r'$\bm{f/f^{*}=25}$',xy=(1.6,2.7),xycoords='data',color='white')

#axs[0].annotate(r'$\mathrm{a)}$',xy=(0.0, 0.75), xycoords='figure fraction')

#axs[0].annotate(r'$\mathrm{b)}$',xy=(0.0, 0.52), xycoords='figure fraction')

plt.savefig('fig4b.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()


