% Physical parameters
nu = 0.3;
E = 2e12;
rho = 7850;

Lx = 0.2;
Ly = 0.3;
Lz = 5e-4;

Nmodes = 30;

ldim = [Lx, Ly, Lz];
BCs  = [1e15 0;
        1e15 0;
        1e15 0;
        1e15 0];

h = 0.01;
[Om,PhiRef,NxRef,NyRef,~,~] = magpie(rho,E,nu,ldim,h,BCs,Nmodes,"none",true);
h = Lx / 30;
[~,Phi,Nx,Ny,~,~] = magpie(rho,E,nu,ldim,h,BCs,Nmodes,"none",true);

save("eigenmac_test.mat", "PhiRef","Phi", "Om","Ny","Nx","NyRef","NxRef")
%% iterpolation

xref=linspace(0,Lx,NxRef+1);
yref=linspace(0,Ly,NyRef+1);
[XREF,YREF]=meshgrid(xref,yref);
x=linspace(0,Lx,Nx+1);
y=linspace(0,Ly,Ny+1);
[X,Y]=meshgrid(x,y);

Qreftemp= PhiRef(:,1) ;
Qreftemp = reshape(Qreftemp,[(NyRef+1),(NxRef+1)]) ;
Spreftemp= interp2(XREF,YREF,Qreftemp,X,Y);



%%
[QMAC,OmMAC] = eigenMAC(PhiRef,NxRef,NyRef,Phi,Nx,Ny,h,Nmodes,Lx,Ly,Om); % Ensures that mode order is consistent

diff_indices = find(OmMAC ~= Om);

display(diff_indices)
%%
