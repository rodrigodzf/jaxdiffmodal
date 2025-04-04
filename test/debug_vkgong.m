addpath(genpath('.'))
Npsi = 10;
Nphi = 10;

Lx = 0.2;
Ly = 0.3;
hd = 0.0005;

nu = 0.3;
E = 2e12;
rho = 7850;

L = Lx;
BC = 'SimplySupported'; % Boundary conditions: 'SimplySupported'


%%
s1 = i1_mat(Npsi,Nphi,Lx);
s2 = i2_mat(Npsi,Nphi,Lx);
s3 = i3_mat(Npsi,Nphi,Lx);
s4 = i4_mat(Npsi,Nphi,Lx);
s5 = i5_mat(Npsi,Nphi,Lx);
s9 = i9_mat(Npsi,Nphi,Lx);
s10 = i10_mat(Npsi,Nphi,Lx);
s11 = i11_mat(Npsi,Nphi,Lx);
s12 = i12_mat(Npsi,Nphi,Lx);
s13 = i13_mat(Npsi,Nphi,Lx);
%%
S = (Npsi^2)/2;
[mode_t] = ComputeTransverseEigenfrequenciesRectangular( BC, Lx, Ly,  Nphi  );

D = E*hd^3/12/(1-nu^2);
om_dim = sqrt(D/rho/hd)*mode_t(:, 4);

m1 = g1(Npsi,Nphi,S,Lx, mode_t);
m2 = g2(Npsi,Nphi,S,Lx, mode_t);

m3 = g3(Npsi,Nphi,S,Ly, mode_t);
m4 = g4(Npsi,Nphi,S,Ly, mode_t);

m5 = g5(Npsi,Nphi,S,Lx, mode_t);
m6 = g6(Npsi,Nphi,S,Ly, mode_t);


%%
[coeff0, coeff1, coeff2, om2] = AiryStressFactorsCalculation(BC, Npsi,Lx,Ly);

% 
om2 = sqrt((D/rho/hd) * om2);

[ H0, H1, H2 ] = H_tensorRectangular( coeff0, coeff1, coeff2, Nphi, Npsi, Lx, Ly, mode_t, BC);

H0 = H0(1:Npsi, 1:Nphi*Nphi);
H1 = H1(1:Npsi, 1:Nphi*Nphi);
H2 = H2(1:Npsi, 1:Nphi*Nphi);

H0 = reshape(H0,[Npsi, Nphi, Nphi]);
H1 = reshape(H1,[Npsi, Nphi, Nphi]);
H2 = reshape(H2,[Npsi, Nphi, Nphi]);

H0_slice = squeeze(H0(1, :, :));

fprintf('vkgong phi omega:     (%s)\n', strjoin(cellstr(num2str(om_dim(1:4))),', '));
fprintf('vkgong psi omega:     (%s)\n', strjoin(cellstr(num2str(om2(1:4))),', '));

save("debug_vkgong.mat","s1","s2","s3","s4","s5","s9","s10","s11","s12","s13","m1", "m2","m3","m4","m5","m6", "coeff0", "coeff1", "coeff2", "H0", "H1", "H2", "om_dim");
