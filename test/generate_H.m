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
Nvec = [50 100];
npts = 2;

for iter = 1:npts

    h = Lx / Nvec(iter); % Defining h
    [Om, Phi, Nx, Ny, ~, ~] = magpie(rho, E, nu, ldim, h, BCsPhi, Nmodes, "none", true);

    if iter == 1
        Phiref = Phi; % Setting a refference for Psi
        Nxref = Nx; % Setting a refference for Nx
        Nyref = Ny; % Setting a refference for Ny
    else
        [Phi, Om] = eigenMAC(Phiref, Nxref, Nyref, Phi, Nx, Ny, h, Nmodes, Lx, Ly, Om);
        Phi = eigensign(Phiref, Nxref, Nyref, Phi, Nx, Ny, h, Nmodes, Lx, Ly);
    end

    [Om2, Psi, ~, ~, ~, zetafourth] = magpie(rho, E, nu, ldim, h, BCsPsi, Nmodes, "none", false); % see magpie doc

    if iter == 1

        Psiref = Psi; % Setting a refference for Psi

    else

        [Psi, Om2] = eigenMAC(Psiref, Nxref, Nyref, Psi, Nx, Ny, h, Nmodes, Lx, Ly, Om2); % Ensures that mode order is consistent

        Psi = eigensign(Psiref, Nxref, Nyref, Psi, Nx, Ny, h, Nmodes, Lx, Ly); % Ensures that the polarization is consistent

    end

    disp(iter)

    zeta = (zetafourth) .^ (1/4);

    Hv = zeros(Nmodes, Nmodes, Nmodes);

    Ev = zeros(Nmodes, Nmodes, Nmodes);




end


Dxx = DxxBuild(Nx, Ny, h);
Dyy = DyyBuild(Nx, Ny, h);
Dxy = DxyBuild(Nx, Ny, h);

% precompute norms
norms_k = zeros(Nmodes, 1);
for k = 1:Nmodes
    Psik = Psi(:, k);
    norms_k(k) = trapzIntcalc(Psik .* Psik, h, Nx, Ny);
end

tic;
for k = 1:Nmodes

    Phik = Phi(:, k); Psik = Psi(:, k);

    for p = 1:Nmodes

        Phip = Phi(:, p);

        for q = p:Nmodes

            Phiq = Phi(:, q); Psiq = Psi(:, q);

            LPhipPhiq = vkOperator(Phip, Phiq, Dxy, Dxx, Dyy);
            
            norm_k = norms_k(k);

            Hv(k, q, p) = trapzIntcalc(Psik .* LPhipPhiq, h, Nx, Ny) / (sqrt(norm_k));
            Hv(k, p, q) = trapzIntcalc(Psik .* LPhipPhiq, h, Nx, Ny) / (sqrt(norm_k)); %Coupling coefficient tensor
        end

    end

end
toc;

% filter very close to zero entries

for ii = 1:size(Hv, 1)

    temp0 = Hv(ii, :);

    v0 = find(abs(temp0 / max(abs(temp0))) < 1e-8);

    temp0(v0) = 0;

    Hv(ii, :) = temp0;

end

Hv_slice = squeeze(Hv(1, :, :));

fprintf('magpie phi omega:     (%s)\n', strjoin(cellstr(num2str(Om(1:4))), ', '));
fprintf('magpie psi omega:     (%s)\n', strjoin(cellstr(num2str(Om2(1:4))), ', '));

% h = 0.001;
% Nx = 100;
% Ny = 150;
% Dxx = DxxBuild(Nx, Ny, h);
% Dyy = DyyBuild(Nx, Ny, h);
% Dxy = DxyBuild(Nx, Ny, h);
% F1 = Phi(:, 1);
% F2 = Phi(:, 2);
% LPhipPhiq = vkOperator(F1, F2, Dxy, Dxx, Dyy);
% 
% save('Hv.mat', 'Hv', 'Phi', 'Psi', 'Dxx', 'Dyy', 'Dxy', 'LPhipPhiq')

%%
% 
% Spreftemp = Phi(:, 1);
% Sptemp = Phi(:, 8);
% h = 0.001;
% a = trapzIntcalc(F1, h, Nx, Ny);
% 
% Q1 = Spreftemp - Sptemp;
% Q1vec = reshape(Q1, [(Ny + 1) * (Nx + 1)], 1);
% Q2 = Spreftemp + Sptemp;
% Q2vec = reshape(Q2, [(Ny + 1) * (Nx + 1)], 1);
% 
% % subplot(2,2,3)
% % mesh(Q1)
% % subplot(2,2,4)
% % mesh(Q2)
% %intq1=trapzIntcalc(Q1vec.*Q1vec,h,Nx,Ny);
% %intq2=trapzIntcalc(Q2vec.*Q2vec,h,Nx,Ny);
% 
% sgn = sign(-abs(trapzIntcalc(Q1vec .* Q1vec, h, Nx, Ny)) + abs(trapzIntcalc(Q2vec .* Q2vec, h, Nx, Ny)));
% 
% ret = Dxx * Spreftemp;
% ret = reshape(ret, [(Ny + 1), (Nx + 1)]);
% % Qtemp   = sgn*Qtemp;
% % Qsign(:,nQ) = Qtemp ;
