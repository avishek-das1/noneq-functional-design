!*****************************************************************************80
!*****************************************************************************80
program gradientdescent 
use mpi
implicit none
!parameters
integer,parameter:: N=7 !Number of particles
double precision, parameter:: sigmaA=1.0d0 !WCA sigma parameter
integer,parameter :: ithreshold=25d5 !number of timesteps to relax into steady state
integer(kind=8),parameter:: steps=2d7+ithreshold !total number of simulation timesteps in each optimization step
double precision,parameter:: dt=5d-5 !timestep
double precision,parameter:: temp=1.0d0 !temperature
double precision,dimension(N,N):: D0AA !matrix of interaction energies, only upper half plane is relevant
double precision, parameter:: alpha=10d0 !Morse potential parameter
double precision,parameter:: ewca=10.0d0 !WCA potential parameter
double precision, parameter:: pi=4.0d0*atan(1.0d0)
double precision, parameter:: rho=0.01 !packing fraction of colloids
double precision, parameter:: L=(N*pi* (sigmaA)**3/(6.0d0*rho))**(1.0d0/3.0d0) !length of box in 3d
double precision:: lambda !biasing parameter in cost function

!derivative parameters
double precision, parameter:: sigmaAp6=sigmaA**6, sigmaAp12=sigmaAp6**2
double precision, parameter:: Awca=2**(1./6) *sigmaA !cutoff for WCA forces
double precision, parameter:: Amin=Awca !position of Morse potential minima
double precision, parameter:: morsecutoffA=Amin+1.0*sigmaA !cutoff for Morse potential
double precision :: frc !parameter for shifted forces approximation
double precision, parameter:: rthick=1.0d0 !rskin-rcut !for neighbor list
double precision, parameter:: rskinA=morsecutoffA+rthick !for neighbor list
double precision, parameter:: prefacAA=2.0d0*alpha !prefactor for morse force

!parameters
double precision,parameter :: etavar=(2.0d0*temp*dt)**0.5d0 !variance of Gaussian noise, friction is assumed to be 1

!data structures
double precision, dimension(N,3):: x0,x1,x2,F1,F1wca,F1morse !holders for position and forces
double precision, dimension(N,3) :: rand1,rand2,r,thetar,eta !holders for random noise
integer, parameter:: deltamax=1d5,deltamin=1d5,deltadiff=1d5 !\Delta t for integrating Malliavin weights
integer, parameter:: deltanum=(deltamax-deltamin)/deltadiff +1 !number of different \Delta t values
double precision, dimension(N,N,3) :: delu !use only the upper half plane !gradient of force on each particle with respect to each design parameter
double precision, dimension(N,N,deltamax+1) :: q !use only the upper half plane !history keeping for Malliavin weight, \zeta in pseudocode
double precision, dimension(N,N) :: delqtemp !difference in old and new Malliavin weight, \Delta\zeta in pseudocode
double precision, dimension(N,N,deltanum) :: delomega2,delomega3 !correlation function part of the gradient, and \overline{\Delta\zeta} in pseudocode
double precision, dimension(N,N,deltanum) :: delomega2dum !for MPI communication
double precision, dimension(N,N):: delomega1,delomega1dum !non-correlation function part of the gradient (g in pseudocode), and data-structure for MPI
double precision, dimension(N,N):: delomega !total gradient

!shear force parameters
double precision::shear !shear flow rate
double precision, dimension(N,3):: F1shear !shear flow force
double precision, dimension(N,3) :: delush !shear force derivative
double precision, dimension(deltamax+1) :: qsh !Malliavin weight for shear
double precision::delqshtemp !difference in old and new Malliavin weights
double precision, dimension(deltanum):: delomega2sh,delomega3sh !Malliavin weight structures
double precision, dimension(deltanum):: delomega2shdum,delomega3shdum !MPI communications
double precision:: delomega1sh,delomega1shdum !Malliavin weight structures
double precision:: delomegash !total gradient
double precision:: tshear,cz,cx,cdelx !parameters for Lee Edwards boundary

integer :: sizerand
integer, allocatable :: putrand(:) !for seeding random numbers
double precision :: omega,omegatemp,omegadum !averaged reward (\overline{\xi}), reward at every step (\xi), and dummy variable for MPI
double precision :: hintav,hintavdum  !average of the flux or current observable
integer, dimension(8) :: values,valuesn !!for timestamp
double precision :: absr
integer :: indicator !!indicates overlap between particles in initial state
double precision, dimension(N,3):: dxnei !cumulative displacement vector, needed for neighbor list
double precision:: drnei,drneimax,drneimax2 !for neighbor list
integer:: nnflag !for neighbor list
integer,dimension(N,N):: nlist !neighbor list array

integer :: irdf !will calculate rdf (radial distribution function) if irdf is 1
integer,parameter :: bins=500
double precision,dimension(bins,2) :: rdf
double precision :: binlen,xr(3) !rdf can only to calculated upto L/2

integer, dimension(N,N):: adjacency !adjacency matrix
double precision, parameter:: adjcutoff=1.348 !bond distance cutoff for defining adjacency
integer:: summax,sumtemp,btot,summin,nummin,nummax !parameters for identifying symmetry of the cluster
integer, parameter:: numhint=500 !number of timesteps for calculating current over (\tau/\delta t in pseudocode)
integer,dimension(numhint,2):: hint !history of indicators of being in either of the two reactive states
integer:: newhp,newhp1,oldhp !pointers for the hint array

!mpi parameters and variables
integer :: ierr,tag,cores,id,my_id,status(MPI_STATUS_SIZE)
double precision :: start,finish
character (len=90) :: filename

!optimizing parameters::
double precision:: graderr, graderrsh !magnitudes of Dij and shear gradients
integer,parameter:: iterations=50 !total number of optimization steps
double precision:: e !damping coefficient for optimization
double precision:: delD0(N,N), delshear !update step for Dij and shear values
integer :: iexit,indfirst !for exiting optimization loop, and for choosing value in current step

!cluster analysis by depth-first search
integer:: clustnum(N) !assigns a cluster label number to each particle 
integer:: maxlabel !carries the total number of clusters
integer:: sizearray(N,3) !array to hold size of each cluster, maximum can be N clusters; second index holds total bonds in each cluster; third index holds the number of bonds formed by the maximally bonded particle in each cluster

!counters
integer(kind=8) :: i,j,k,p,s,index,itervar
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
open(unit=38,file='omega.txt')
open(unit=40,file='D0running.txt')
open(unit=42,file='init.txt')
open(unit=44,file='shear.txt')

D0AA=0.0d0
do i=1,N
    do j=i+1,N
        !read(33,*) D0AA(i,j)
        D0AA(i,j)=5.0d0!D0AA(i,j)/10.0d0*4.0d0
        D0AA(j,i)=D0AA(i,j)
    end do
end do

!do i=1,NA
!    read(42,*) x0(i,:)
!end do
x0=0.0d0

end if

!broadcast coefficients
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(D0AA,N*N,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)

shear=40.0d0

!broadcast initial state
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(x0,N*3,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)

!potential loop over lambda should go here
lambda=1d14
delD0=0.0
delshear=0.0d0
iexit=0
indfirst=0
graderr=1000
graderrsh=1000

!!optimization iteration
do itervar=1,iterations
start=MPI_Wtime()

putrand(:)=1d6*my_id!+walkerno !seed random_number
call random_seed(put=putrand)

!optimization step
D0AA=D0AA+delD0
shear=shear+10*delshear

do i=1,N
    do j=1,N
        if (D0AA(i,j)<0) then
            D0AA(i,j)=0
            !D0AA(j,i)=D0AA(i,j)
        end if
        if (D0AA(i,j)>10) then
            D0AA(i,j)=10
            !D0AA(j,i)=D0AA(i,j)
        end if
    end do
end do

if (shear<0.0d0) shear=0.0d0
if (shear>50.0d0) shear=50.0d0

!print *, my_id,D0AA
!call flush()

if (my_id==0) then !receive gradients from other walkers
omega=0.0d0
hintav=0.0d0
delomega1=0.0d0
delomega2=0.0d0
delomega1sh=0.0d0
delomega2sh=0.0d0
do i=1,cores-1 !total no. of mpi messages to recv
id=i
call MPI_Recv(omegadum,1,MPI_DOUBLE_PRECISION,id,int(100*i+1),MPI_COMM_WORLD,status,ierr)
omega=omega+omegadum
call MPI_Recv(delomega1dum,N*N,MPI_DOUBLE_PRECISION,id,int(100*i+2),MPI_COMM_WORLD,status,ierr)
delomega1=delomega1+delomega1dum
call MPI_Recv(delomega2dum,N*N*deltanum,MPI_DOUBLE_PRECISION,id,int(100*i+3),MPI_COMM_WORLD,status,ierr)
delomega2=delomega2+delomega2dum
call MPI_Recv(hintavdum,1,MPI_DOUBLE_PRECISION,id,int(100*i+4),MPI_COMM_WORLD,status,ierr)
hintav=hintav+hintavdum
call MPI_Recv(delomega1shdum,1,MPI_DOUBLE_PRECISION,id,int(100*i+5),MPI_COMM_WORLD,status,ierr)
delomega1sh=delomega1sh+delomega1shdum
call MPI_Recv(delomega2shdum,deltanum,MPI_DOUBLE_PRECISION,id,int(100*i+6),MPI_COMM_WORLD,status,ierr)
delomega2sh=delomega2sh+delomega2shdum
end do
hintav=hintav/(cores-1)
omega=omega/(cores-1)
delomega1=delomega1/(cores-1)
delomega2=delomega2/(cores-1)
delomega1sh=delomega1sh/(cores-1)
delomega2sh=delomega2sh/(cores-1)

else !other walkers running trajectories

!print *, my_id, D0AA
!call flush()

!initialization for rdf calculation
binlen=1.5/bins!L/(2.0d0*bins)
irdf=0
rdf=0.0d0

hint=0

!prefacAA=2.0d0*alpha
!force shift at cutoff
frc=(exp(-alpha*(morsecutoffA-Amin))-1.0d0)*exp(-alpha*(morsecutoffA-Amin))

!Unform random initial placement of particles respective volume exclusion
x0(1,:)=0.0d0 !first particle in the middle of the box
do i=2,N
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

!print *, x0

x2=x0

!initializing averages as 0
omega=0.0d0
hintav=0.0d0
dxnei=1000.0d0
delomega1(:,:)=0.0d0
delomega2(:,:,:)=0.0d0
delomega3(:,:,:)=0.0d0
delomega1sh=0.0d0
delomega2sh=0.0d0
delomega3sh=0.0d0
q(:,:,:)=0.0d0
qsh=0.0d0

!pointers for q history
newq=deltamax+1
do p=1,deltanum
    oldq(p)=newq-deltamin-(p-1)*deltadiff
end do

!pointers for indicator history
hint=0
newhp=numhint
oldhp=1

tshear=0.0d0
if (my_id==0)print *, L

call date_and_time(VALUES=values)
!if (my_id==1) then
!do j=1,NA
!write(*,*) x0(j,:)
!!call flush()
!end do
!end if

nnflag=1

!trajectory loop
do i=1,steps

    x1=x2

    do j=1,N
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
    do j=1,N
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
&   N,sigmaA,alpha,ewca,&
&   sigmaAp6,sigmaAp12,&
&   Awca, Amin, &
&   morsecutoffA,shear,&
&   D0AA,delu,delush,frc,&
&   prefacAA,rskinA,&
&   nlist,nnflag,&
&   tshear,adjcutoff,adjacency)

    if (i>ithreshold-deltamax) then
        newq1=mod(newq,deltamax+1)+1

        q(:,:,newq1)=q(:,:,newq)
        do j=1,N !rows
            do k=j+1,N !columns
                do jj=1,1 !multi !This functionality can be invoked to have multiple copies of the same alphabet along with dimension of D0AA = N/multi
                    do kk=1,1 !multi
                        jjlabel=(jj-1)*N+j
                        kklabel=(kk-1)*N+k
                        q(j,k,newq1)=q(j,k,newq1)+(etavar*dot_product(eta(jjlabel,:)-eta(kklabel,:),&
&       delu(jjlabel,kklabel,:))/(2.0d0*temp)) !works if deluold(jjlabel,kklabel,:) is the force on jjlabel due to kklabel
                    end do
                end do
            end do
        end do

        !do j=1,N !rows
        !    k=j
        !    !do k=j+1,Nu !columns
        !        do jj=1,multi
        !            do kk=jj+1,multi
        !                jjlabel=(jj-1)*Nu+j
        !                kklabel=(kk-1)*Nu+k
        !                q(j,k,newq1)=q(j,k,newq1)+(etavar*dot_product(eta(jjlabel,:)-eta(kklabel,:),&
!&       delu(jjlabel,kklabel,:))/(2.0d0*temp)) !works if deluold(jjlabel,kklabel,:) is the force on jjlabel due to kklabel
        !            end do
        !        end do
        !    !end do
        !end do

        qsh(newq1)=qsh(newq)
        ! differs from actual formula because eta here has variance 1, not
        ! 2*temp*dt !!
        do j=1,N
            do k=j+1,N
              xr=delush(j,:)-delush(k,:)
              xr(1)=xr(1)-nint(xr(1)/L)*L !depends on the z so follows pbc of z, but acts in x direction
              qsh(newq1)=qsh(newq1)+etavar*(eta(j,1)-eta(k,1))*(xr(1))/(2.0d0*temp*N) !low-variance estimate at small particle density
            end do
        end do

        newq=newq1
        do p=1,deltanum
            oldq(p)=mod(oldq(p),deltamax+1)+1
        end do
    end if

!!  Calculate rewards and gradients
    if (i>ithreshold-numhint) then  !start indicator calculation
            newhp1=mod(newhp,numhint)+1
        !if (mod(i-ithreshold,omegastep)==0) then !re-evaluate hint
        !edit the adjacency based on new cutoff
            btot=0
            hint(newhp1,:)=0
            do j=1,N
                do k=j+1,N
                    if (adjacency(j,k)==0) cycle
                    btot=btot+1 !total number of bonds
                end do
            end do

        !now check geometry of the cluster
            if (btot==15) then !maximally bonded cluster cluster
                summin=10
                nummin=0
                summax=0
                nummax=0
                do j=1,N
                    sumtemp=sum(adjacency(j,:))
                    if (sumtemp<summin) then
                        summin=sumtemp
                        nummin=1
                    else if (sumtemp==summin) then
                        nummin=nummin+1
                    end if
                    if (sumtemp>summax) then
                        summax=sumtemp
                        nummax=1
                    else if (sumtemp==summax) then
                        nummax=nummax+1
                    end if
                end do
                !if (summax==6 .and. summin==3 .and. nummax==1 .and. nummin==2) hint(newhp1,1)=1 !this is a C2 structure
                !if (summax==6 .and. nummax==2) hint(newhp1,2)=1 !this is a C2v structure
                if (summin==3 .and. nummin==3) hint(newhp1,1)=1 !this is a C3v_1 structure
                if (summax==5 .and. summin==3) hint(newhp1,2)=1 !this is a C3v_2 structure
            end if !end condition on btot
            newhp=newhp1
            oldhp=mod(oldhp,numhint)+1 
        !end if
      end if
      
      if (i>ithreshold) then !start omega calculation
        omegatemp=0.0d0
        omegatemp=omegatemp+lambda*(hint(newhp,2)*hint(oldhp,1))!-hint(newhp,1)*hint(oldhp,2)) !probability flow from c3v_1 to c3v_2
        do j=1,N
            omegatemp=omegatemp-dot_product(F1morse(j,:)+F1shear(j,:),&
            &   F1morse(j,:)+F1shear(j,:))/(4.0d0*temp)
            delomega1sh=delomega1sh-(dot_product(F1morse(j,:)+F1shear(j,:),delush(j,:)))/(2.0d0*temp)                
        end do
        
        do j=1,N
            do k=j+1,N !choose one parameter
                do jj=1,1 !multi !This functionality can be invoked to have multiple copies of the same alphabet along with dimension of D0AA = N/multi
                    do kk=1,1 !multi
                        jjlabel=(jj-1)*N+j
                        kklabel=(kk-1)*N+k
                        delomega1(j,k)=delomega1(j,k) &
&   -dot_product(F1morse(jjlabel,:)+F1shear(jjlabel,:)&
&   -F1morse(kklabel,:)-F1shear(kklabel,:),&
&   delu(jjlabel,kklabel,:))/(2.0d0*temp)
                    end do
                end do
            end do
        end do

        !do j=1,Nu
        !    k=j
        !    !do k=j,Nu !choose one parameter
        !        do jj=1,multi
        !            do kk=jj+1,multi
        !                jjlabel=(jj-1)*Nu+j
        !                kklabel=(kk-1)*Nu+k
        !                delomega1(j,k)=delomega1(j,k) &
!&   -dot_product(F1morse(jjlabel,:)+F1shear(jjlabel,:)&
!&   -F1morse(kklabel,:)-F1shear(kklabel,:),&
!&   delu(jjlabel,kklabel,:))/(2.0d0*temp)
        !            end do
        !        end do
        !    !end do
        !end do

        omega=omega+omegatemp
        hintav=hintav+hint(newhp,2)*hint(oldhp,1)!-hint(newhp,2)*hint(oldhp,1)
        do p=1,deltanum
            delqtemp=q(:,:,newq)-q(:,:,oldq(p))
            delomega2(:,:,p)=delomega2(:,:,p)+omegatemp*delqtemp
            delomega3(:,:,p)=delomega3(:,:,p)+delqtemp
            delqshtemp=qsh(newq)-qsh(oldq(p))
            delomega2sh(p)=delomega2sh(p)+omegatemp*delqshtemp
            delomega3sh(p)=delomega3sh(p)+delqshtemp
        end do
    end if

    x2=x1+dt*(F1+F1wca+F1morse+F1shear)+etavar *eta !Langevin equation for propagation
    dxnei=dxnei+x2-x1

    tshear=tshear+dt
    !Lees-Edwards periodic boundary conditions
    do j=1,N
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

do i=1,bins
rdf(i,1)=(i-0.5d0)*binlen
rdf(i,2)=rdf(i,2)* L**3 /((N*(N-1)*4.0d0*pi*rdf(i,1)**2 *binlen*(steps-ithreshold)/2.0d0))
!write(33,*) rdf(i,:)
end do

id=0
call MPI_Send(omega,1,MPI_DOUBLE_PRECISION,id,int(100*my_id+1),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega1,N*N,MPI_DOUBLE_PRECISION,id,int(100*my_id+2),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega2,N*N*deltanum,MPI_DOUBLE_PRECISION,id,int(100*my_id+3),MPI_COMM_WORLD,ierr)
call MPI_Send(hintav,1,MPI_DOUBLE_PRECISION,id,int(100*my_id+4),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega1sh,1,MPI_DOUBLE_PRECISION,id,int(100*my_id+5),MPI_COMM_WORLD,ierr)
call MPI_Send(delomega2sh,deltanum,MPI_DOUBLE_PRECISION,id,int(100*my_id+6),MPI_COMM_WORLD,ierr)

end if !for the trajectory

!!at this point master node has all the averaged gradients

!!broadcast the averaged gradients
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(omega,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(hintav,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega1,N*N,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega2,N*N,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
delomega=delomega1+delomega2(:,:,deltanum)

call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega1sh,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega2sh,deltanum,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
delomegash=delomega1sh+delomega2sh(deltanum)

graderr=(sum(delomega)**2)**0.5
graderrsh=abs(delomegash)

do i=1,N
    do j=i+1,N
        delomega(j,i)=delomega(i,j)
    end do
end do

if (my_id==0) then
!print out the running output
write(*,*)'lambda and omega are', lambda,omega,hintav
write(*,*) 'D0AA is', D0AA
write(*,*) 'Shear is', shear
write(*,*) 'graderr and graderrsh are', graderr, graderrsh
write(38,*) omega,hintav
do i=1,N
    do j=i+1,N
        write(40,*) D0AA(i,j)
    end do
end do
write(44,*) shear

call flush()
finish=MPI_Wtime()
print *, 'Time taken is',finish-start
end if

if (indfirst==0) then
    e=0.4/maxval(abs(delomega))
    indfirst=1
end if

!The following line can be uncommented by defining a tolerance value for convergence
!if (graderr<tol .and. graderrsh<tol) iexit=1

!!gradient descent update step
delD0=e*delomega
delshear=e*delomegash

if (iexit==1) exit !exit from the trajectory

end do !the while loop that optimizes.

call MPI_Finalize(ierr)

end program gradientdescent
!*****************************************************************************80
!*****************************************************************************80
!*****************************************************************************80
!force computations at every time step
subroutine force(x,&
&   L,F,Fwca,Fmorse,Fshear,&
&   NA,sigmaA,alpha,ewca,&
&   sigmaAp6,sigmaAp12,&
&   Awca, Amin, &
&   morsecutoffA,shear,&
&   D0AA,delu,delush,frc,&
&   prefacAA,rskinA,&
&   nlist,nnflag,&
&   tshear,adjcutoff,adjacency) !bins, irdf,rdf,binlen could be passed from main program to here for rdf computation
implicit none

integer :: i,j,k,NA
double precision:: x(NA,3),L,F(NA,3),Fwca(NA,3),Fmorse(NA,3),Fshear(NA,3),tshear
double precision:: sigmaA,alpha,ewca
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
!put 1body force here for F
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
Fint=24.0d0*ewca*(2.0d0*sigmaAp12/rprod2-sigmaAp6/rprod1)*rcap
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
Fint=24.0d0*ewca*(2.0d0*sigmaAp12/rprod2-sigmaAp6/rprod1)*rcap
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
subroutine clustsearch(adjacency,NA,clustnum,maxlabel,sizearray) !depth-first search for cluster analysis
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
recursive subroutine labelassign(NA,adjacency,i,j,clustnum,sizearray) !recursive assigning of cluster labels to all particles
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

