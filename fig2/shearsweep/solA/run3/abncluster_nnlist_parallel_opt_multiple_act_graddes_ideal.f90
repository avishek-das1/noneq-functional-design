!*****************************************************************************80
!*****************************************************************************80
program srk2 !(Mx,My,Mz,M2,a,b,delomega1,delomega2,delomega1int,delomega2int,kldf,omega,walkerno,my_id,N,xold)
use mpi
implicit none
!parameters
integer,parameter:: Nu=6,multi=1    !Multiple copies of the system
integer,parameter:: N=Nu*multi,NA=N
integer,parameter:: NB=N-NA
double precision, parameter:: sigmaA=1.0d0
integer,parameter :: ithreshold=25d5!0.625d6 !time it takes to equilibrate
integer(kind=8),parameter:: steps=2d7+ithreshold!1d8+ithreshold
double precision,parameter:: tau=5d-5 !timestep never more than this
double precision,parameter:: D=1.0d0 !temperature
double precision,dimension(NA,NA):: D0AA !matrix of interaction energies, only upper half plane is relevant
double precision,dimension(Nu,Nu):: D0uu !matrix of variational parameters, only upper half plane is relevant
double precision, parameter:: alpha=10d0 !Morse potential parameters
double precision,parameter:: epsilon=10.0d0
double precision, parameter:: pi=4.0d0*atan(1.0d0)
double precision, parameter:: rho=0.01 !A density
double precision, parameter:: L=(NA*pi* (sigmaA)**3/(6.0d0*rho))**(1.0d0/3.0d0)
double precision:: kldf

!derivative parameters
double precision, parameter:: sigmaAp6=sigmaA**6, sigmaAp12=sigmaAp6**2
double precision, parameter:: Awca=2**(1./6) *sigmaA
double precision, parameter:: Amin=Awca
double precision, parameter:: morsecutoffA=Amin+1.0*sigmaA
double precision :: frc !keeps changing for changing D0AA
!double precision, parameter:: urc=-0.00045398899185673634, frc=-0.004539786860886241 !for 5kbt
!double precision, parameter:: urc=-0.0009079779837134727, frc=-0.009079573721772483 !for 10kbt
double precision, parameter:: rthick=1.0d0 !rskin-rcut
double precision, parameter:: rskinA=morsecutoffA+rthick
double precision, parameter:: prefacAA=2.0d0*alpha !prefactor for morse force

!parameters
!double precision,parameter:: driftx=1.0d0,drifty=0.0d0,driftz=0.0d0
double precision,parameter :: etavar=(2.0d0*D*tau)**0.5d0

!data structures
double precision, dimension(NA,3):: x0,x1,x2,F1,F1wca,F1morse
double precision, dimension(NA,3) :: rand1,rand2,r,thetar,eta
double precision, dimension(NA,3) :: rand12,rand22,r2,thetar2
!integer, parameter:: deltamax=1,deltamin=1,deltadiff=1
integer, parameter:: deltamax=1d5,deltamin=1d5,deltadiff=1d5
integer, parameter:: deltanum=(deltamax-deltamin)/deltadiff +1
double precision, dimension(NA,NA,3) :: delu !use only the upper half plane
double precision, dimension(Nu,Nu,deltamax+1) :: q !use only the upper half plane
double precision, dimension(Nu,Nu) :: delqtemp
double precision, dimension(Nu,Nu,deltanum) :: delomega2,delomega3
double precision, dimension(Nu,Nu,deltanum) :: delomega2dum
double precision, dimension(Nu,Nu):: delomega1,delomega1dum
integer, parameter:: omegastep=1 !hint is reevaluated every omegastep steps

double precision:: actparams(Nu),delactparams(Nu),actparamstemp(Nu),graderract
double precision,parameter:: etaactvar=(6.0d0*D*tau)**0.5d0
double precision:: costheta(NA),sintheta(NA),cosphi(NA),sinphi(NA),etatheta(NA),etaphi(NA)
double precision:: diract(NA,3),F1act(NA,3),act(NA)
!the role of delush is taken by diract
double precision:: qact(Nu,deltamax+1),delqacttemp(Nu),delomega1act(Nu),delomega1actdum(Nu)
double precision,dimension(Nu,deltanum):: delomega2act,delomega2actdum,delomega3act
double precision:: delomegaact(Nu)

!shear force parameters
double precision::shear,delshear,sheartemp,graderrsh
double precision, dimension(NA,3):: F1shear
double precision, dimension(NA,3) :: delush !shear force derivative
double precision, dimension(deltamax+1) :: qsh
double precision::delqshtemp
double precision, dimension(deltanum):: delomega2sh,delomega3sh
double precision, dimension(deltanum):: delomega2shdum,delomega3shdum
double precision:: delomega1sh,delomega1shdum
double precision:: delomegash
double precision:: tshear,cz,cx,cdelx

double precision :: grnd
integer :: sizerand
integer, allocatable :: putrand(:)
double precision :: omega,omegatemp,omegadum,hintav(4),hintavdum(4)
!double precision :: qtemp, qtempint
integer, dimension(8) :: values,valuesn !!for timestamp
double precision :: absr
integer :: indicator !!indicates overlap between particles in initial state
integer:: posstep,poscount
double precision, dimension(NA,3):: dxnei
double precision:: drnei,drneimax,drneimax2
integer:: nnflag
integer,dimension(NA,NA):: nlist

integer :: irdf !will calculate rdf if irdf is 1
integer,parameter :: bins=500
double precision,dimension(bins,2) :: rdf
double precision :: binlen,xr(3) !rdf can only to calculated upto L/2

integer, dimension(NA,NA):: adjacency
double precision, parameter:: adjcutoff=1.348 !adjacency cutoff
!double precision:: hint(2),hintold(2)
integer:: summax,sumtemp,btot
integer, parameter:: numhint=500 !number of timesteps for calculating current over
integer,dimension(numhint,2):: hint
integer:: newhp,newhp1,oldhp !pointers for the hint array

!mpi parameters and variables
!!integer,parameter :: Nw=63 !must be integer multiple of cores !no fopenmp for now
integer :: ierr,tag,cores,id,my_id,status(MPI_STATUS_SIZE)
double precision :: start,finish
character (len=90) :: filename

!optimizing parameters::
double precision:: graderr,tempelement
double precision, parameter:: tol=5d-3, gamma=0.0d0
double precision:: e !damping coefficient
double precision:: delD0(Nu,Nu)
double precision, allocatable :: xold(:,:,:) !xold functionality not yet included
integer :: iexit,indfirst
double precision:: delomega (Nu,Nu)
double precision:: D0uutemp(Nu,Nu)
integer:: clustnum(NA) !assigns a cluster label number to each particle 
integer:: maxlabel !carries the total number of clusters
integer:: sizearray(NA,3) !array to hold size of each cluster, maximum can be NA clusters; second index holds total bonds in each cluster; third index holds the number of bonds formed by the maximally bonded particle in each cluster
integer, dimension(2) :: temploc
!double precision,parameter:: e_adam=1d-6,alpha_adam=1.0,beta1=0.9,beta2=0.999
!double precision:: m_adam(Nu,Nu),v_adam(Nu,Nu),m_shear,v_shear !adam parameters
!integer:: t_adam

!counters
integer(kind=8) :: i,j,k,p,s,shearcount,index
integer :: newq, newq1
integer, dimension(deltanum) :: oldq
integer:: jj,kk,jjlabel,kklabel

call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD,my_id,ierr)
call MPI_Comm_size(MPI_COMM_WORLD,cores,ierr)

!initialize random numbers here
call random_seed(size=sizerand)
allocate(putrand(sizerand))

if (my_id==0) then
open(unit=33,file='D0AA.txt')
!open(unit=330,file='acts.txt')
!open(unit=331,file='delomega1.txt')
!open(unit=34,file='delomega2.txt')
open(unit=38,file='omega.txt')
!open(unit=39,file='traj.txt')
open(unit=40,file='D0running.txt')
!open(unit=400,file='actsrunning.txt')
!open(unit=41,file='result.txt')
open(unit=42,file='init.txt')
!open(unit=43,file='delD0.txt')
open(unit=44,file='shear.txt')

D0uu=0.0d0
do i=1,Nu
    do j=i+1,Nu
        read(33,*) D0uu(i,j)
!        D0uu(i,j)=3.0d0
        D0uu(j,i)=D0uu(i,j)
    end do
end do

!do i=1,NA
!    read(42,*) x0(i,:)
!end do
x0=0.0d0

actparams=0.0d0
!do i=1,Nu
!   read(330,*) actparams(i)
!end do

end if

!broadcast coefficients
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(D0uu,Nu*Nu,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(actparams,Nu,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
D0uutemp=D0uu
actparamstemp=actparams

shear=47.2263
sheartemp=shear

do i=1,multi !copy the adjacency matrix for every block
    do j=1,multi
        D0AA((i-1)*Nu+1:i*Nu,(j-1)*Nu+1:j*Nu)=D0uu(:,:)
    end do
end do

do i=1,multi
    act((i-1)*Nu+1:i*Nu)=actparams
end do

!broadcast initial state
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(x0,NA*3,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)

!potential loop over kldf should go here
kldf=1d14
delD0=0.0
delactparams=0.0d0
delshear=0.0d0
iexit=0
indfirst=0
graderr=1000
graderrsh=1000
graderract=1000

putrand(:)=1d6*my_id+3!+walkerno
call random_seed(put=putrand)

!!put a while loop here that ensures error is less than tolerance
do shearcount=0,70,5
start=MPI_Wtime()

D0uu=D0uu!+delD0
shear=shearcount!shear!+delshear
actparams=actparams!+10.0d0*delactparams

do i=1,Nu
    do j=1,Nu
        if (D0uu(i,j)<0) then
            D0uu(i,j)=0
            !D0AA(j,i)=D0AA(i,j)
        end if
        if (D0uu(i,j)>10) then
            D0uu(i,j)=10
            !D0AA(j,i)=D0AA(i,j)
        end if
    end do
end do
do i=1,multi
    do j=1,multi
        D0AA((i-1)*Nu+1:i*Nu,(j-1)*Nu+1:j*Nu)=D0uu(:,:)
    end do
end do
D0uutemp=D0uu

!if (shear<0.0d0) shear=0.0d0
!if (shear>50.0d0) shear=50.0d0
sheartemp=shear

do i=1,Nu
    if (actparams(i)<0) actparams(i)=0
    if (actparams(i)>50) actparams(i)=50
end do
do i=1,multi
    act((i-1)*Nu+1:i*Nu)=actparams
end do
actparamstemp=actparams

!print *, my_id,D0AA
!call flush()

if (my_id==0) then !receive gradients from other walkers
omega=0.0d0
hintav=0.0d0
delomega1=0.0d0
delomega2=0.0d0
delomega1sh=0.0d0
delomega2sh=0.0d0
delomega1act=0.0d0
delomega2act=0.0d0
do i=1,cores-1 !total no. of mpi messages to recv
id=i
call MPI_Recv(omegadum,1,MPI_DOUBLE_PRECISION,id,int(100*i+1),MPI_COMM_WORLD,status,ierr)
omega=omega+omegadum
call MPI_Recv(delomega1dum,NA*NA,MPI_DOUBLE_PRECISION,id,int(100*i+2),MPI_COMM_WORLD,status,ierr)
delomega1=delomega1+delomega1dum
call MPI_Recv(delomega2dum,NA*NA*deltanum,MPI_DOUBLE_PRECISION,id,int(100*i+3),MPI_COMM_WORLD,status,ierr)
delomega2=delomega2+delomega2dum
call MPI_Recv(hintavdum,4,MPI_DOUBLE_PRECISION,id,int(100*i+4),MPI_COMM_WORLD,status,ierr)
hintav=hintav+hintavdum
call MPI_Recv(delomega1shdum,1,MPI_DOUBLE_PRECISION,id,int(100*i+5),MPI_COMM_WORLD,status,ierr)
delomega1sh=delomega1sh+delomega1shdum
call MPI_Recv(delomega2shdum,deltanum,MPI_DOUBLE_PRECISION,id,int(100*i+6),MPI_COMM_WORLD,status,ierr)
delomega2sh=delomega2sh+delomega2shdum
call MPI_Recv(delomega1actdum,NA,MPI_DOUBLE_PRECISION,id,int(100*i+7),MPI_COMM_WORLD,status,ierr)
delomega1act=delomega1act+delomega1actdum
call MPI_Recv(delomega2actdum,NA*deltanum,MPI_DOUBLE_PRECISION,id,int(100*i+8),MPI_COMM_WORLD,status,ierr)
delomega2act=delomega2act+delomega2actdum
end do
omega=omega/(cores-1)
hintav=hintav/(cores-1)
delomega1=delomega1/(cores-1)
delomega2=delomega2/(cores-1)
delomega1sh=delomega1sh/(cores-1)
delomega2sh=delomega2sh/(cores-1)
delomega1act=delomega1act/(cores-1)
delomega2act=delomega2act/(cores-1)

else !other walkers running trajectories

!print *, my_id, D0AA
!call flush()

binlen=1.5/bins!L/(2.0d0*bins)
irdf=0
rdf=0.0d0
hint=0
!prefacAA=2.0d0*alpha
frc=(exp(-alpha*(morsecutoffA-Amin))-1.0d0)*exp(-alpha*(morsecutoffA-Amin))

!L=(N*pi*lambda**3/(6.0d0*rho) )**(1.0d0/3.0d0)
!ithreshold=1d5
!steps=1d7+ithreshold
!!Uniformly random placement of particles respecting volume exclusion
!if (xold(1,1)<0) then !make new initial conditions

x0(1,:)=0.0d0!0.5*L !first particle in the middle of the box
do i=2,NA
!    print *, 'starting particle', i
    indicator=1
    do while (indicator==1)
        call random_number(rand1(i,:)) !three random coordinates for ith particle
                rand1(i,:)=(rand1(i,:)-0.5d0)*L !rescales the coordinates to the whole box
        indicator=0
        do j=1,i-1  !Check overlap with previous particles
            xr=rand1(i,:)-x0(j,:)
            !!periodic boundaries
            xr(1)=xr(1)-L*nint(xr(1)/L)
            xr(2)=xr(2)-L*nint(xr(2)/L)
            xr(3)=xr(3)-L*nint(xr(3)/L)
            absr=dot_product(xr,xr)**0.5
            if (absr<sigmaA) indicator=1  !overlap!
        end do
    end do
    x0(i,:)=rand1(i,:)
!    print *, 'ending particle', i
end do
!ithreshold=1d6
!steps=1d7+ithreshold
!else
!x0=xold
!end if

!print *, x0

x2=x0

costheta=0.0d0
sintheta=1.0d0
cosphi=1.0d0
sinphi=0.0d0
diract(:,1)=sintheta*cosphi
diract(:,2)=sintheta*sinphi
diract(:,3)=costheta

poscount=0
posstep=100
dxnei=1000.0d0
delomega1(:,:)=0.0d0
delomega2(:,:,:)=0.0d0
delomega3(:,:,:)=0.0d0
delomega1sh=0.0d0
delomega2sh=0.0d0
delomega3sh=0.0d0
delomega1act=0.0d0
delomega2act=0.0d0
delomega3act=0.0d0
q(:,:,:)=0.0d0
qsh=0.0d0
qact=0.0d0
newq=deltamax+1
do p=1,deltanum
    oldq(p)=newq-deltamin-(p-1)*deltadiff
end do
hint=0
newhp=numhint
oldhp=1
omega=0.0d0
hintav=0.0d0
tshear=0.0d0
print *, L

!call date_and_time(VALUES=values)
if (my_id==1) then
do j=1,NA
write(*,*) x0(j,:)
!call flush()
end do
end if

!initial values of force
nnflag=1
!    call force(x2,&
!&   L,F1,F1wca,F1morse,F1shear,&
!&   NA,sigmaA,alpha,epsilon,&
!&   sigmaAp6,sigmaAp12,&
!&   Awca, Amin, &
!&   morsecutoffA,shear,&
!&   D0AA,delu,delush,frc,&
!&   prefacAA,rskinA,&
!&   nlist,nnflag,&
!&   tshear,adjcutoff,adjacency)
!
!    do j=1,NA    
!        F1act(j,:)=act(j)*diract(j,:)
!    end do

!trajectory loop
do i=1,steps

    x1=x2

    do j=1,NA
        if (mod(i,2)==1) then
            call random_number(rand1(j,:))
            call random_number(rand2(j,:))
            r(j,:)=(-2.0d0*log(rand2(j,:)))**0.5d0
            thetar(j,:)=2.0d0*pi*rand1(j,:)
            rand1(j,:)=r(j,:)*cos(thetar(j,:))
            rand2(j,:)=r(j,:)*sin(thetar(j,:))
            eta(j,:)=rand1(j,:)
        else
            eta(j,:)=rand2(j,:)
        end if
    end do

!Updating neighbor list if required by calculating two maximum displacements
    drneimax=0.0
    drneimax2=0.0
    do j=1,NA
        drnei=sqrt(dot_product(dxnei(j,:),dxnei(j,:)))
        if (drnei > drneimax) then
            drneimax2=drneimax
            drneimax=drnei
        else
            if (drnei > drneimax2) then
                drneimax2=drnei
            endif
        endif
    enddo
    if (drneimax+drneimax2 > rthick) then
        nnflag=1 !update neighbor list in the next call for force
    !print *,'Updating neighbor list at step', i
    end if

    if (nnflag==1) dxnei=0

    call force(x1,&
&   L,F1,F1wca,F1morse,F1shear,&
&   NA,sigmaA,alpha,epsilon,&
&   sigmaAp6,sigmaAp12,&
&   Awca, Amin, &
&   morsecutoffA,shear,&
&   D0AA,delu,delush,frc,&
&   prefacAA,rskinA,&
&   nlist,nnflag,&
&   tshear,adjcutoff,adjacency)

!!  Calculate delomega2
    if (i>ithreshold-numhint) then  !start hint calculation
            newhp1=mod(newhp,numhint)+1
        !if (mod(i-ithreshold,omegastep)==0) then !re-evaluate hint
        !edit the adjacency based on new cutoff
            btot=0
            hint(newhp1,:)=0
            do j=1,NA
                do k=j+1,NA
                    if (adjacency(j,k)==0) cycle
                    btot=btot+1
                end do
            end do

        !now check whether the correct Oh cluster
            if (btot==12) then !at least big cluster
                summax=0
                do j=1,NA
                    sumtemp=sum(adjacency(j,:))
                    if (sumtemp>summax) then
                        summax=sumtemp
                    end if
                end do
                if (summax==4) hint(newhp1,1)=1 !octahedron
                if (summax==5) hint(newhp1,2)=1 !polytetrahedron
            end if !end condition on btot
            newhp=newhp1
            oldhp=mod(oldhp,numhint)+1 
        !end if
      end if
      
      if (i>ithreshold) then !start omega calculation
        omegatemp=0.0d0
        omegatemp=omegatemp+kldf*(hint(newhp,1)*hint(oldhp,2)-hint(newhp,2)*hint(oldhp,1)) !current flowing from polytd to oct

        omega=omega+omegatemp
	hintav(1)=hintav(1)+hint(newhp1,2)
	hintav(2)=hintav(2)+hint(newhp1,1)
        hintav(3)=hintav(3)+hint(newhp,1)*hint(oldhp,2)!-hint(newhp,2)*hint(oldhp,1)
        hintav(4)=hintav(4)+hint(newhp,2)*hint(oldhp,1)!-hint(newhp,2)*hint(oldhp,1)

    end if

    x2=x1+tau*(F1+F1wca+F1morse+F1shear+F1act)+etavar *eta
    dxnei=dxnei+x2-x1

    tshear=tshear+tau
    do j=1,NA
        cz=nint(x2(j,3)/L)
        cx=x2(j,1)-cz*shear*L*tshear
        x2(j,1)=cx-L*nint(cx/L)
        x2(j,2)=x2(j,2)-L*nint(x2(j,2)/L)
        x2(j,3)=x2(j,3)-L*cz
    end do

end do
print *, 'walker',my_id,'done'

!print *, omega
omega=omega/((steps-ithreshold))
hintav=hintav/((steps-ithreshold))
!print *, omega
delomega1=delomega1/((steps-ithreshold))
delomega2=delomega2/((steps-ithreshold))
delomega3=delomega3/((steps-ithreshold))
delomega2=delomega2-omega*delomega3
delomega1sh=delomega1sh/((steps-ithreshold))
delomega2sh=delomega2sh/((steps-ithreshold))
delomega3sh=delomega3sh/((steps-ithreshold))
delomega2sh=delomega2sh-omega*delomega3sh
delomega1act=delomega1act/(steps-ithreshold)
delomega2act=delomega2act/(steps-ithreshold)
delomega3act=delomega3act/(steps-ithreshold)
delomega2act=delomega2act-omega*delomega3act

do i=1,bins
rdf(i,1)=(i-0.5d0)*binlen
rdf(i,2)=rdf(i,2)* L**3 /((NA*(NA-1)*4.0d0*pi*rdf(i,1)**2 *binlen*(steps-ithreshold)/2.0d0))
!write(33,*) rdf(i,:)
end do

id=0
call MPI_Send(omega,1,MPI_DOUBLE_PRECISION,id,int(100*my_id+1),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega1,Nu*Nu,MPI_DOUBLE_PRECISION,id,int(100*my_id+2),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega2,Nu*Nu*deltanum,MPI_DOUBLE_PRECISION,id,int(100*my_id+3),MPI_COMM_WORLD,ierr)
call MPI_Send(hintav,4,MPI_DOUBLE_PRECISION,id,int(100*my_id+4),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega1sh,1,MPI_DOUBLE_PRECISION,id,int(100*my_id+5),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega2sh,deltanum,MPI_DOUBLE_PRECISION,id,int(100*my_id+6),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega1act,Nu,MPI_DOUBLE_PRECISION,id,int(100*my_id+7),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega2act,Nu*deltanum,MPI_DOUBLE_PRECISION,id,int(100*my_id+8),MPI_COMM_WORLD,ierr)

end if !for the trajectory

!!at this point master node has all the averaged gradients

!!broadcast the averaged gradients
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(omega,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(hintav,4,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega1,Nu*Nu,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega2,Nu*Nu,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
delomega=delomega1+delomega2(:,:,deltanum)

call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega1sh,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega2sh,deltanum,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
delomegash=delomega1sh+delomega2sh(deltanum)

call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega1act,Nu,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega2act,Nu*deltanum,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
delomegaact=delomega1act+delomega2act(:,deltanum)

graderr=(sum(delomega)**2)**0.5
graderrsh=abs(delomegash)
graderract=(sum(delomegaact)**2)**0.5
do i=1,Nu
    do j=i+1,Nu
        delomega(j,i)=delomega(i,j)
    end do
end do

if (my_id==0) then
!print out the running output
write(*,*)'kldf and omega are', kldf,omega,hintav
write(*,*) 'D0uu is', D0uu
write(*,*) 'Shear is', shear
write(*,*) 'actparams is', actparams
write(*,*) 'graderr and graderrsh and graderract are', graderr, graderrsh, graderract
write(38,*) hintav(:)
do i=1,Nu
    do j=i+1,Nu
        write(40,*) D0uu(i,j)
    end do
end do
!do i=1,Nu
!    write(400,*) actparams(i)
!end do
write(44,*) shear

call flush()
finish=MPI_Wtime()
print *, 'Time taken is',finish-start
end if

!do k=1,Nu
!    if (my_id==0)   print *, 'Diagonal term is',delomega(k,k)
!    delomega(k,k)=0.0d0
!end do

end do !the while loop that optimizes.

call MPI_Finalize(ierr)

end program srk2
!*****************************************************************************80
!*****************************************************************************80
!*****************************************************************************80
subroutine force(x,&
&   L,F,Fwca,Fmorse,Fshear,&
&   NA,sigmaA,alpha,epsilon,&
&   sigmaAp6,sigmaAp12,&
&   Awca, Amin, &
&   morsecutoffA,shear,&
&   D0AA,delu,delush,frc,&
&   prefacAA,rskinA,&
&   nlist,nnflag,&
&   tshear,adjcutoff,adjacency)
implicit none

integer :: i,j,k,NA
double precision:: x(NA,3),L,F(NA,3),Fwca(NA,3),Fmorse(NA,3),Fshear(NA,3),tshear
double precision:: sigmaA,alpha,epsilon
double precision:: sigmaAp6,sigmaAp12
double precision:: Awca,Amin
double precision:: morsecutoffA,shear,frc
double precision:: D0AA(NA,NA), delu(NA,NA,3),delush(NA,3)
double precision:: prefacAA
double precision:: rskinA
integer:: nnflag, nlist(NA,NA)
integer:: index
integer, parameter:: bins=10,irdf=0
double precision:: rdf(bins),binlen
double precision :: expfac,xr(3),absr,rcap(3),Fint(3),rprod1,rprod2
double precision:: cz,cdelx
double precision:: adjcutoff
integer:: adjacency(NA,NA)

F=0.0d0
Fwca=0.0d0
Fmorse=0.0d0
Fshear=0.d0
delu=0.0d0
delush=0.0d0
adjacency=0

if (nnflag==0) then !do calculation according to neighbor list

do i=1,NA
!put 1body force here
delush(i,1)=x(i,3) !shear is in x direction and depends on z position
Fshear(i,1)=shear*x(i,3)

!do j=i+1,N
do k=1,nlist(i,1)
  j=nlist(i,k+1)
  if (j<=i) cycle

!!define absr and xr right away and work with those
xr=x(i,:)-x(j,:)
!!periodic boundaries
cz=nint(xr(3)/L)
cdelx=xr(1)-cz*shear*L*tshear
xr(1)=cdelx-nint(cdelx/L)*L
xr(2)=xr(2)-nint(xr(2)/L)*L
xr(3)=xr(3)-cz*L
absr=dot_product(xr,xr)**0.5
rcap=xr/absr
!!rdf evaluation
!if (irdf==1) then
!index=floor(abs(r)/binlen)
!rdf(index+1)=rdf(index+1)+1
!end if

if (irdf==1) then
    !print *, 'reaching here1'
    !print *, absr
    index=floor(absr/binlen)
    if (index+1<=bins) then 
        !print *, index+1,bins
        rdf(index+1)=rdf(index+1)+1
        !print *, 'reaching here2'
    end if
end if

!!WCA evaluation
if (absr<Awca) then
rprod1=absr*absr
rprod1=rprod1*rprod1
rprod1=rprod1*rprod1/absr !7th power
rprod2=rprod1*rprod1/absr !13th power
Fint=24.0d0*epsilon*(2.0d0*sigmaAp12/rprod2-sigmaAp6/rprod1)*rcap
Fwca(i,:)=Fwca(i,:)+Fint
Fwca(j,:)=Fwca(j,:)-Fint
end if
!!Morse potential evaluation
if (absr<morsecutoffA) then
expfac=exp(-alpha*(absr-Amin))
Fint=prefacAA*((expfac-1.0d0)*expfac-frc)*rcap
delu(i,j,:)=Fint
delu(j,i,:)=-Fint
Fmorse(i,:)=Fmorse(i,:)+D0AA(i,j)*Fint
Fmorse(j,:)=Fmorse(j,:)-D0AA(i,j)*Fint
if (absr<adjcutoff) then
adjacency(i,j)=1
adjacency(j,i)=1
end if
end if

end do

end do

else !make new neighbor list
nlist(:,1)=0

do i=1,NA
!put 1body force here
delush(i,1)=x(i,3) !shear is in x direction and depends on z position
Fshear(i,1)=shear*x(i,3)

do j=i+1,NA

!!define absr and xr right away and work with those
xr=x(i,:)-x(j,:)
!!periodic boundaries
cz=nint(xr(3)/L)
cdelx=xr(1)-cz*shear*L*tshear
xr(1)=cdelx-nint(cdelx/L)*L
xr(2)=xr(2)-nint(xr(2)/L)*L
xr(3)=xr(3)-cz*L
absr=dot_product(xr,xr)**0.5
rcap=xr/absr
!!rdf evaluation
!if (irdf==1) then
!index=floor(abs(r)/binlen)
!rdf(index+1)=rdf(index+1)+1
!end if

if (irdf==1) then
    !print *, 'reaching here1'
    !print *, absr
    index=floor(absr/binlen)
    if (index+1<=bins) then 
        !print *, index+1,bins
        rdf(index+1)=rdf(index+1)+1
        !print *, 'reaching here2'
    end if
end if

if (absr<=rskinA) then !i and j are neighbors
nlist(i,nlist(i,1)+2)=j
nlist(j,nlist(j,1)+2)=i
nlist(i,1)=nlist(i,1)+1
nlist(j,1)=nlist(j,1)+1

!!WCA evaluation
if (absr<Awca) then
rprod1=absr*absr
rprod1=rprod1*rprod1
rprod1=rprod1*rprod1/absr !7th power
rprod2=rprod1*rprod1/absr !13th power
Fint=24.0d0*epsilon*(2.0d0*sigmaAp12/rprod2-sigmaAp6/rprod1)*rcap
Fwca(i,:)=Fwca(i,:)+Fint
Fwca(j,:)=Fwca(j,:)-Fint
end if
!!Morse potential evaluation
if (absr<morsecutoffA) then
expfac=exp(-alpha*(absr-Amin))
Fint=prefacAA*((expfac-1.0d0)*expfac-frc)*rcap
delu(i,j,:)=Fint
delu(j,i,:)=-Fint
Fmorse(i,:)=Fmorse(i,:)+D0AA(i,j)*Fint
Fmorse(j,:)=Fmorse(j,:)-D0AA(i,j)*Fint
if (absr<adjcutoff) then
adjacency(i,j)=1
adjacency(j,i)=1
end if

end if

end if !end calculation for i and j being neighbors

end do !loop over 2body forces

end do !loop over 1body forces
nnflag=0
end if

end subroutine force
!*****************************************************************************80
!*****************************************************************************80
!************************************************************************
!*****************************************************************************80
subroutine clustsearch(adjacency,NA,clustnum,maxlabel,sizearray)
implicit none
integer,intent(in) :: NA,adjacency(NA,NA)
integer:: clustnum(NA)
integer:: i,j,maxlabel,sizearray(NA,3) !maxlabel carries the maximum used cluster label till now

do i=1,NA
    if (clustnum(i)>0) then 
        cycle !this particle already has been assigned
    else
        !print *,i,maxlabel+1
        clustnum(i)=maxlabel+1
        maxlabel=maxlabel+1
        sizearray(maxlabel,1)=sizearray(maxlabel,1)+1 !size of the cluster increases
        sizearray(maxlabel,3)=sum(adjacency(i,:)) !maximum bonded particle in this cluster is the first one
    end if
    
    !assign the label to all children
    do j=1,NA
        if (adjacency(i,j)==1) then !assign the child
            if (clustnum(j)<0) then
                call labelassign(NA,adjacency,i,j,clustnum,sizearray)
            end if
            sizearray(maxlabel,2)=sizearray(maxlabel,2)+1 !new bond
        end if
    end do
end do
sizearray(:,2)=sizearray(:,2)/2
!the atom with maximum number of bonds can be found outside. Also the total number of bonds in each cluster can be found outside.But it will be expensive for big systems.
end subroutine clustsearch
!!*******************************************************************************************!!
!!*******************************************************************************************!!
!!*******************************************************************************************!!
!!*******************************************************************************************!!
!this subroutine assigns the label of parent i to self j and all its children
recursive subroutine labelassign(NA,adjacency,i,j,clustnum,sizearray)
implicit none
integer, intent(in):: NA, adjacency(NA,NA),i,j
integer:: clustnum(NA),sizearray(NA,3)

integer:: k,p

!!assign parents label to self
clustnum(j)=clustnum(i)
sizearray(clustnum(i),1)=sizearray(clustnum(i),1)+1 !cluster size has increased
!print *,j,clustnum(i),i
sizearray(clustnum(i),3)=max(sum(adjacency(j,:)),sizearray(clustnum(i),3)) !recheck if new particle is the maximal bonded particle

!!propagate the label to all unlabelled children
do k=1,NA
    if (adjacency(j,k)==1) then
        if (clustnum(k)<0) then
            call labelassign(NA,adjacency,j,k,clustnum,sizearray)
        end if
        sizearray(clustnum(i),2)=sizearray(clustnum(i),2)+1 !new bond
    end if
end do
end subroutine labelassign
!!*******************************************************************************************!!
!!*******************************************************************************************!!
!!*******************************************************************************************!!
!!*******************************************************************************************!!

