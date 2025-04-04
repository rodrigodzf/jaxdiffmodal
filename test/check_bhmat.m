addpath ./private/magpie

nu = 0.3;
E = 2e12;
rho = 7850;

Lx = 0.2;
Ly = 0.3;
Lz = 0.0005;

Nmodes = 10;

BCsPhi = [1e15 0e15;
           1e15 0e15;
           1e15 0e15;
           1e15 0e15];

BCsPsi = [1e15 1e15;
           1e15 1e15;
           1e15 1e15;
           1e15 1e15];

ldim = [Lx Ly Lz];

%%--- derived parameters (don't change here)
D = E * Lz^3 / 12 / (1-nu^2);
Nx      = 100;
Ny      = 150;
h = 0.002;
%% Build Biharmonic
biHarm = bhmat(BCsPhi,[Nx Ny], h, Lz, E, nu);

[Q,Dm] = eigs(biHarm,Nmodes,'smallestabs') ;
[~,indSort] = sort(diag((Dm))) ;
Q = Q(:,indSort) ;

Dm    = diag(Dm) ;
Om    = sqrt(abs(Dm))*sqrt(D/rho/Lz) ;

shouldNormalise = true;
if shouldNormalise
    for nQ = 1 : Nmodes
        Qtemp   = Q(:,nQ) ;
        Qnorm   = trapzIntcalc(Qtemp.*Qtemp,h,Nx,Ny);
        Qnorm = sqrt(Qnorm);
        Qtemp   = Qtemp /  Qnorm;
        Q(:,nQ) = Qtemp ;
    end
end


save('bhmat.mat', 'biHarm');