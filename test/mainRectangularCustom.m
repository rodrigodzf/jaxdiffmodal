%% This main file is used to execute the simulation functions for the nonlinear vibrations of thin plates. 
clear all;
clc;
%% Input parameters files
PlateCharacteristicsFileName = 'PlateCharacteristicsRectangular.mat'; % Physical characteristics of the plate: Dimensions, imperfection profile, material and boundary conditions.  
SimulationParametersFileName = 'SimulationParametersRectangular.mat'; % Parameters related to the simulation: Time length, scheme, number of modes, output points, accuracy.
GammaFileName = 'GammaRectangular.mat'; % Name of the file containing the Gamma Tensor.  
ScoreFileName = 'ScoreParametersRectangular.mat'; % Characteristics of the excitation. 
OutputFileName = 'ResultsRectangular'; % Name of the results files and folder. 

%% Characteristics

Lx = 0.2; % Plate dimension 1
Ly = 0.3; % Plate dimension 2
hd = 0.0005; % Plate thickness

% Imperfection characteristics
H = 0; % Imperfection height
ImperfectionType = '2DRaisedCosine'; 
xWidth = Lx/2;
yWidth = Ly/3;
proj = [];
modeIndices = [];
ModeType = 'All';

error_coef = 0.10; % Admitted error from 0 to 1.

% Material parameters
nu = 0.3;
E = 2e12;
rho = 7850;

% Damping % X = {'Undamped', 'PowerLaw'}
X = [];
c = zeros(1500, 1);

BC = 'SimplySupported'; % Boundary conditions: 'SimplySupported'

PlateCharacteristicsFileName = 'CustomCharParRect.mat';
save('Parameters/Input files/CustomCharParRect.mat', "Lx", "Ly", "hd", "H", "ImperfectionType", "xWidth", "yWidth", "proj", "modeIndices", "ModeType", "error_coef", "nu", "E", "rho", "X", "c", "BC");

%% Simulation parameters 
Nphi = 10; %select number of transverse modes
Npsi = 10; %transverse modes
NA = 0; %Number of modes considered to compute the eigenfrequencies of the imperfect plate.

% Time simulation parameters
scheme = 'verlet'; %% Integration scheme: "ECS" or "verlet"
fsd = 44100; %% Sampling frequency
Tsd = 2; %Simulation time

% Model parameters
Nx = 20; % Number of discretization points for r
Ny = 30; % Number of discretization points for \theta

% Output points
op = [0.1 0.1];
SimulationParametersFileName = 'CustomSimParRect.mat';
save('Parameters/Input files/CustomSimParRect.mat', "op", "Nx", "Ny", "NA", "Nphi", "Npsi", "scheme", "fsd", "Tsd");
%%

[Lx, Ly, hd, E, rho, BC, e,  Nphi, Npsi, scheme, H0, H1, H2, filename, Ai, C, C1, C2, kx, ky, om_dim, rp, tad, fsd, Tsd] = plate_def_rect(PlateCharacteristicsFileName, SimulationParametersFileName, OutputFileName, GammaFileName );

%%
[ f_time, Tn ] = score_rect( ScoreFileName, Lx, Ly, hd, rho, kx, ky, BC, Nphi, scheme, C, fsd, Tsd);
save('debug.mat',"f_time", "rp", "H1");
%% Time simulation
switch scheme
    case 'ECS'
        [ out ] = ftime_imperfect_ECS( Nphi, Npsi, Ai, H0, H1, H2, C, C1,C2, Tn, e, f_time, rp);
        
    case 'verlet'
        [ out ] = ftime_imperfect_verlet( Nphi, Npsi, Ai, H1, C, C1, C2, Tn, e, f_time, rp);
    
    otherwise
        disp('Unknown scheme');
end

%% Save results
out_vel = diff(out,1,1)*fsd; % Output velocity
save(sprintf('%s.mat',filename), 'out', 'out_vel',  'fsd');

%% Generate audio from output (Comment this block if the code is not used for sound synthesis purposes)
for i = 1:size(rp,1)
    disp('what')
    disp(i)
    audiowrite(sprintf('%s-Op_%d.wav',filename, i),out_vel(:,i)/1.1/max(abs(out_vel(:,i))),fsd);
end
