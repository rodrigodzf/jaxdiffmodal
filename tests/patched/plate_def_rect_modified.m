function [Lx, Ly, hd, E, rho, BC, e, Nphi, Npsi, scheme, H0, H1, H2, filename, Ai, C, C1, C2, kx, ky, om_dim, rp, tnd, fsd, Tsd, c] = plate_def_rect_modified(PlateCharacteristicsFileName, SimulationParametersFileName, OutputFileName, GammaFileName, X )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                                                                   %
%                              VK-Gong                              % 
%                                                                   %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function parses the files indicated in the main script in order to 
% define the plate and create the necessary variables for the code.

%% Plate characteristics
if exist(PlateCharacteristicsFileName, 'file') == 2
    load(PlateCharacteristicsFileName);
else
    disp('Plate characteristics file not found. Using preset instead.');
    % Geometrical characteristics
    Lx = 0.2; % Plate dimension 1
    Ly = 0.3; % Plate dimension 2
    hd = .8e-3; % Plate thickness
    
    % Imperfection characteristics
    H = 1e-2; % Imperfection height
    ImperfectionType = '2DRaisedCosine'; 
    xWidth = Lx/2;
    yWidth = Ly/3;
    proj = [];
    modeIndices = [];
    
    error_coef = 0.10; % Admitted error from 0 to 1.
    
    % Material parameters
    nu = 0.38;
    E = 2e11;
    rho = 7860;

    % Damping % X = {'Undamped', 'PowerLaw'}
    X = [];
    c = zeros(1500, 1);
        
    % Boundary conditions parameters
    BC = 'SimplySupported'; % Boundary conditions: 'SimplySupported'



end

e = (Lx*Ly/4)/rho*E;
tnd = 1;

%% Simulation parameters
if exist(SimulationParametersFileName, 'file') == 2
    load(SimulationParametersFileName);
else
    disp('Simulation parameters file not found. Using preset instead.');
    
    % Accuracy parameters
    Nphi = 10; %select number of transverse modes
    Npsi = 3; %transverse modes
    NA = 10; %Number of modes considered to compute the eigenfrequencies of the imperfect plate.
    
    % Time simulation parameters
    scheme = 'ECS'; %% Integration scheme: "ECS" or "verlet"
    fsd = 0; %% Sampling frequency
    Tsd = 10; %Simulation time
    
    % Model parameters
    Nx = 400; % Number of discretization points for r
    Ny = 500; % Number of discretization points for \theta
    
    % Output points
    op = [0.1 0.1];
    

end

% Integration steps
dx = Lx/Nx;
dy = Ly/Ny;

Nop =  size(op,1); % Number of output points



%% H coefficients and mode files loading
[mode_t] = ComputeTransverseEigenfrequenciesRectangular( BC, Lx, Ly,  Nphi  );

disp('Transverse eigenfrequencies computed');

D = E*hd^3/12/(1-nu^2);
om_dim = sqrt(D/rho/hd)*mode_t(:, 4);
kx = mode_t(:, 2);
ky = mode_t(:, 3);

[H0, H1, H2] = LoadHTensorRectangular(BC, Nphi, Npsi, Lx, Ly, mode_t);

disp('H tensors loaded');

%% Output file parameters
mkdir(OutputFileName);
filename = sprintf('%s/%s_%s_%s', OutputFileName,OutputFileName, scheme, BC);

%% Imperfection modelling
if isempty(proj)
    if H == 0
        Ai = zeros(Nphi,1); % Projection coefficients
        proj = [];
        modeIndices = [];

    else

        [ ~,~,~,proj,modeIndices,~ ]  = ProjectionOfTheImperfectionRectangular( H, Lx, Ly, Nx, Ny, BC, error_coef, ModeType, ImperfectionType, xWidth, yWidth, Nphi);
        Ai = zeros(Nphi,1); % Projection coefficients
        Ai(modeIndices)=proj;
        Ai=Ai(1:Nphi); %%In case that Nphi<max(modeIndices)
        
        disp('Performed computation of projection coefficients');
    end
end
save(sprintf('%s-Imperfection_Parameters.mat',filename),'Ai','modeIndices', 'proj');


%% Computation of eigenfrequencies of the imperfect plate
if NA ~= 0
    GammaFileName = sprintf('Parameters/H files/Rectangular/%s',GammaFileName);
    if exist(GammaFileName, 'file') ~= 2
        disp('Gamma file not found. Creating it');
        GammaTensor( H1, GammaFileName, NA, Npsi );
    end

    [Omega, ~] = ComputeEigenfrequenciesImperfectPlate(proj,NA, modeIndices,  om_dim, e, GammaFileName );
    
    disp('Computed eigenfrequencies of the imperfect plate');

    save (sprintf('%s-Eigenfrequencies-Curvature.mat',filename),'Omega');
end

%% sampling rate (Hz),
fsd_lim = om_dim(Nphi); %limit frequency

if (fsd < fsd_lim)
    disp(['Warning: Input sampling rate is below the stability limit. New fsd = ', num2str(fsd_lim)]);
    fsd = fsd_lim;
end

fsd = round(fsd); %sampling rate


%% damping ratios
if ~isempty(X)
    [c] = c_preset(X,om_dim(1:Nphi),Nphi, dFac, dExp, dCons);
end

%timestep
k = 1/fsd;

%% Initialise constant matrices for main loop
C = (1/k^2 + c(1:Nphi)/k/(1));
C1 = zeros(Nphi,1);
C2 = zeros(Nphi,1);

switch scheme 
    case 'ECS' %% Energy Conserving Scheme
        C = diag(C);

        C1 = (-2/k^2 + (om_dim(1:Nphi)/(1)).^2);

        C2 = (1/k^2 - c(1:Nphi)/k/(1));

    case 'verlet' % Stormer - verlet
        C1 = (-2/k^2 + (om_dim(1:Nphi)/(1)).^2)./C;

        C2 = (1/k^2 - c(1:Nphi)/k/(1))./C;

    otherwise
        disp('Unknown scheme');
end

switch BC
    case {'SimplySupported'}
        %% Compute rp factors applied to input 
        rp = zeros(Nop,Nphi);

        for ii = 1 : Nphi
            rp(:, ii) = sin(kx(ii)*pi*op(:,1)/Lx)*sin(ky(ii)*pi*op(:,2)/Ly); %project output

        end

    
    otherwise
        disp('Unknown boundary conditions');

end
        


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp ('Plate input files parsed');

end

