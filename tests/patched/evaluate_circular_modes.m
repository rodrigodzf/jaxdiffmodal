function [weights] = evaluate_circular_modes(mode_t, nu, KR, op, BC, R, dr)
% EVALUATE_CIRCULAR_MODES Evaluate circular plate eigenfunctions at given point
%
% This function computes the modal weights (eigenfunction values) for all
% circular plate modes at a specified radial and angular coordinate.
%
% Inputs:
%   mode_t - Mode table with columns [Index, xkn, k, n, c, xkn^2] where:
%            Index: mode number (1-based)
%            xkn: square root of eigenfrequency
%            k: number of nodal diameters
%            n: number of nodal circles
%            c: configuration (1=cos, 2=sin)
%            xkn^2: eigenfrequency
%   nu     - Poisson ratio
%   KR     - Normalized rotational stiffness (KR = Kr/D)
%   op     - Output point [theta, r] where:
%            theta: angular coordinate (radians)
%            r: radial coordinate (normalized, 0 <= r <= 1)
%   BC     - Boundary condition: 'free', 'elastic', or 'clamped'
%   R      - Plate radius (for normalization)
%   dr     - Integration step for normalization computation
%
% Output:
%   weights - Normalized modal weights at the specified point (Nphi x 1 vector)
%
% Example:
%   % Evaluate all modes at a single point
%   op = [pi/4, 0.7];  % theta = pi/4, r = 0.7
%   weights = evaluate_circular_modes(mode_t, 0.3, inf, op, 'clamped', 1.0, 0.01);

% Input validation
if length(op) ~= 2
    error('op must be a vector with 2 elements [theta, r]');
end

% Extract coordinates  
theta = op(1);
r = op(2);

% Extract mode parameters from mode_t
Nphi = size(mode_t, 1);
c_t = mode_t(:, 5);     % Configuration (cos/sin)
k_t = mode_t(:, 3);     % Number of nodal diameters
xkn = mode_t(:, 2);     % Square root of eigenfrequency

% Initialize output
weights = zeros(Nphi, 1);

% Create radial grid for normalization (shared across all modes)
rr = 0:dr:R;
rr(1) = 1e-10;  % Avoid singularity at r=0

% Loop over all modes (following plate_def_circ.m pattern)
for ii = 1:Nphi
    k = k_t(ii);
    c = c_t(ii);
    xkn_ii = xkn(ii);

    % Compute normalization factor (from norm_modes.m)
    % Compute mode shape over the full domain for normalization
    JJ0_norm = besselj(k, xkn_ii*rr);
    II0_norm = besseli(k, xkn_ii*rr);


    % Compute mode shape at the specific point
    JJ0 = besselj(k, xkn_ii * r);
    II0 = besseli(k, xkn_ii * r);

    % Compute boundary condition coefficients based on BC type
    if strcmp(BC, 'clamped')
        % Clamped case - compute Bessel functions at boundary
        Jkn = besselj(k, xkn_ii);
        Ikn = besseli(k, xkn_ii);

        % Radial part for normalization
        Rkn_norm = JJ0_norm*Ikn - Jkn*II0_norm;
        Rkn = JJ0*Ikn - Jkn*II0;

    elseif strcmp(BC, 'free') || strcmp(BC, 'elastic')
        % Free/elastic case - compute Bessel functions at the boundary
        J0 = besselj(k, xkn_ii);
        J1 = besselj(k-1, xkn_ii);  
        J2 = besselj(k-2, xkn_ii);

        I0 = besseli(k, xkn_ii);
        I1 = besseli(k-1, xkn_ii);
        I2 = besseli(k-2, xkn_ii);

        % Compute boundary condition coefficients
        Jtild = xkn_ii^2*J2 + ((nu-2*k+1)*xkn_ii + KR)*J1 + (k*(k+1)*(1-nu)-KR*k)*J0;
        Itild = xkn_ii^2*I2 + ((nu-2*k+1)*xkn_ii + KR)*I1 + (k*(k+1)*(1-nu)-KR*k)*I0;

        % Radial part for normalization
        Rkn_norm = JJ0_norm - (Jtild*II0_norm/Itild);

        Rkn = JJ0 - (Jtild*II0/Itild);

    end

    % Compute normalization integral
    rR2 = rr .* Rkn_norm.^2;
    Kkn = sqrt(1./trapz(rr, rR2));
    % Apply angular normalization
    if k == 0
        Kkn = Kkn/sqrt(2*pi);
    else
        Kkn = Kkn/sqrt(pi);
    end

    % Angular part (from plate_def_circ.m: cos(k*theta - (c-1)/2*pi))
    angular_part = cos(k * theta - (c-1)/2*pi);

    % Apply normalization to the result
    weights(ii) = Kkn * Rkn * angular_part;
end

end