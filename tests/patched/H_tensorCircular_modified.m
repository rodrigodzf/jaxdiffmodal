function [ H0, H1, H2 ] = H_tensorCircular_modified( Nphi, Npsi, BC, nu, KR, KT, dr_H, mode_t, mode_l )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                                                                   %
%                              VK-Gong                              % 
%                                                                   %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function computes the H matrices for a circular plate of radius 
% equal to unity with the boundary conditions stated by BC. 
% 
% MODIFIED VERSION: Accepts mode_t and mode_l as input parameters instead 
% of loading from files.

%% Input parameters: 
% Nphi: Number of transverse modes
% Npsi: Number of in-plane modes
% BC: Boundary conditions at the edge: 'free', 'clamped', 'elastic'
% nu: Poisson coefficient
% KR: Normalized rotational stiffness KR = Kr/D, with Kr standing for the
% distributed rotational stiffness and D for the bending stiffness of the
% plate. 
% KT: Normalized translational stiffness KT = Kt/D, with Kt standing for
% the distributed rational stiffness at the edge. 
% dr_H: Integration step for the computation of H. 
% mode_t: Transverse modes array [Index, xi, k, n, c, xi^2]
% mode_l: In-plane modes array [Index, zeta, k, n, c, zeta^2]

%% Output parameters
% H0 = H^i_pq
% H1 = H0/zeta_i^2
% H2 = H0/zeta_i^4

% H(i,p,q)
%
% i : Index of in-plane mode i
% p : Index of transverse mode p
% q : Index of transverse mode q

% Set boundary condition parameters (for consistency with original)
switch BC
    case 'free'
        KR = 0;
    case 'clamped'
        KR = inf;
    case 'elastic'
        % Use provided KR and KT values
    otherwise
        error('Unknown boundary conditions: %s', BC);
end

% Validate input arrays
if size(mode_t, 1) < Nphi
    error('mode_t must have at least %d rows for Nphi=%d', Nphi, Nphi);
end
if size(mode_l, 1) < Npsi
    error('mode_l must have at least %d rows for Npsi=%d', Npsi, Npsi);
end

% Transverse modes
k_t = mode_t(1:Nphi,3); % Number of nodal diameters
%n_t = mode_t(1:Nmax,4);
c_t = mode_t(1:Nphi,5); % cos(1)/sin(2) mode
xi_t = mode_t(1:Nphi, 2); 

% In-plane modes
k_l = mode_l(1:Npsi,3); % Number of nodal diameters
%n_l = mode_l(1:Npsi,4);
c_l = mode_l(1:Npsi,5); % cos(1)/sin(2) mode
zeta_l = mode_l(1:Npsi,2);

H0 = zeros(Npsi, Nphi,Nphi);
H1 = zeros(Npsi, Nphi,Nphi);
H2 = zeros(Npsi, Nphi,Nphi);

tic

for p = 1:Nphi
    for q = p:Nphi
        for i = 1:Npsi
            
            H0(i,p,q) = HcoefficientCircular( k_t(p), k_t(q), c_t(p), c_t(q), xi_t(p), xi_t(q), k_l(i), c_l(i), zeta_l(i), nu, KR, dr_H );
            
            H0(i,q,p) = H0(i,p,q);
            
            H1(i,p,q) = H0(i,p,q)/zeta_l(i)^2;
            
            H1(i,q,p) = H0(i,q,p)/zeta_l(i)^2;
            
            H2(i,p,q) = H0(i,p,q)/zeta_l(i)^4;
            
            H2(i,q,p) = H0(i,q,p)/zeta_l(i)^4;
        
        end      
        
    end
end

toc

end