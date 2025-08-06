%% This main file is used to execute the simulation functions for the nonlinear vibrations of circular plates.
%% This is a reference script that generates comprehensive test data for circular membrane analysis.
clear all;
clc;

%% Add paths to VKGong functions
addpath(genpath('../third_party/VKGong/matlab/VK-Gong/Code'));
addpath(genpath('Parameters'));
addpath(genpath('patched'));

%% Input parameters files
GammaFileName = 'GammaCircular.mat'; % Name of the file containing the Gamma Tensor
OutputFileName = 'ResultsCircular'; % Name of the results files and folder

%% Plate Characteristics

% Geometric parameters
Rd = 0.2; % Plate radius in meters
hd = 0.8e-3; % Plate thickness in meters

% Imperfection characteristics
H = 0; % Imperfection height
ImperfectionType = 'RaisedCosine'; % Type of imperfection
rWidth = Rd/2; % Radial width of imperfection
proj = []; % Projection coefficients (empty for no imperfection)
modeIndices = []; % Mode indices for imperfection
ModeType = 'All'; % Type of modes to consider

error_coef = 0.10; % Admitted error from 0 to 1

% Material parameters
nu = 0.38; % Poisson ratio
E = 2e11; % Young's modulus in Pa
rho = 7860; % Density in kg/m³

% Damping parameters
X = 'PowerLaw';
c = zeros(1500, 1); % Damping coefficients
dFac = 0.001; % Damping factor
dExp = 1; % Damping exponent
dCons = 0; % Frequency independent damping constant

% Boundary conditions
BC = 'clamped'; % Boundary conditions: 'free', 'clamped', 'elastic'
KR = inf; % Rotational stiffness (0 for free, inf for clamped, finite for elastic)

% Create parameter directories if they don't exist
if ~exist('Parameters', 'dir')
    mkdir('Parameters');
end
if ~exist('Parameters/Input files', 'dir')
    mkdir('Parameters/Input files');
end
if ~exist('Parameters/Mode files', 'dir')
    mkdir('Parameters/Mode files');
end
if ~exist('Parameters/Mode files/Circular', 'dir')
    mkdir('Parameters/Mode files/Circular');
end
if ~exist('Parameters/H files', 'dir')
    mkdir('Parameters/H files');
end
if ~exist('Parameters/H files/Circular', 'dir')
    mkdir('Parameters/H files/Circular');
end

% Save plate characteristics
PlateCharacteristicsFileName = 'Parameters/Input files/CustomCharParCirc.mat';
save('Parameters/Input files/CustomCharParCirc.mat', ...
    "Rd", "hd", "H", "ImperfectionType", "rWidth", "proj", "modeIndices", ...
    "ModeType", "error_coef", "nu", "E", "rho", "X", "c", "BC", "KR", "dFac", "dExp", "dCons");

%% Simulation parameters
Nphi = 10; % Number of transverse modes
Npsi = 10; % Number of in-plane modes
NA = 0; % Number of modes considered to compute eigenfrequencies of imperfect plate

% Time simulation parameters
scheme = 'verlet'; % Integration scheme: "ECS" or "verlet"
fsd = 44100; % Sampling frequency in Hz
Tsd = 2; % Simulation time in seconds

% Model parameters
Nr = 400; % Number of discretization points for r
Nth = 500; % Number of discretization points for \theta
dr_H = 1e-4; % Integration step used for computing the H coefficients

% Output points [theta, r]
op = [0.5192 0.8962];

% Save simulation parameters
SimulationParametersFileName = 'Parameters/Input files/CustomSimParCirc.mat';
save('Parameters/Input files/CustomSimParCirc.mat', ...
    "op", "Nr", "Nth", "dr_H", "NA", "Nphi", "Npsi", "scheme", "fsd", "Tsd");

%% Generate Reference Data - Core Analysis

fprintf('=== CIRCULAR PLATE REFERENCE DATA GENERATION ===\n');
fprintf('Radius: %.3f m, Thickness: %.4f m\n', Rd, hd);
fprintf('Boundary condition: %s\n', BC);
fprintf('Modes: Nphi=%d, Npsi=%d\n', Nphi, Npsi);

%% 1. Plate Definition and Modal Analysis
fprintf('\n1. Computing plate definition and modal characteristics...\n');
try
    [Rd_out, hd_out, E_out, BC_out, e, Nphi_out, Npsi_out, scheme_out, ...
        H0, H1, H2, filename, Ai, C, C1, C2, k_t, c_t, xkn, JJ, II, Kkn, ...
        rp, tnd, fs, Tsd_out] = plate_def_circ(PlateCharacteristicsFileName, ...
        SimulationParametersFileName, ...
        OutputFileName, GammaFileName);
    
    fprintf('Plate definition completed successfully\n');
    fprintf('   - Transverse modes computed: %d\n', length(k_t));
    fprintf('   - In-plane modes computed: %d\n', Npsi_out);
    fprintf('   - H tensor dimensions: %dx%dx%d\n', size(H0));
    
    % Save modal data
    save('reference_data/circular_reference_modal_data.mat', 'Rd_out', 'hd_out', 'E_out', ...
        'BC_out', 'e', 'k_t', 'c_t', 'xkn', 'JJ', 'II', 'Kkn', 'rp', ...
        'H0', 'H1', 'H2', 'Ai', 'C', 'C1', 'C2');
    
catch ME
    fprintf('Error in plate definition: %s\n', ME.message);
    return;
end

%% 2. Modal Shape Evaluation at Output Points
fprintf('\n2. Evaluating modal shapes at output points...\n');
try
    % Create mode table for evaluate_circular_modes function
    % Format: [Index, xkn, k, n, c, xkn^2]
    mode_t = zeros(length(k_t), 6);
    for i = 1:length(k_t)
        mode_t(i, :) = [i, xkn(i), k_t(i), 1, c_t(i), xkn(i)^2];
    end
    
    % Evaluate modal weights at each output point
    dr = Rd / Nr;  % Radial discretization step
    modal_weights = zeros(size(op, 1), length(k_t));
    for i = 1:size(op, 1)
        weights = evaluate_circular_modes(mode_t, nu, KR, op, BC, Rd, dr);
        modal_weights(i, :) = weights';
        fprintf('   Point %d [θ=%.3f, r=%.3f]: Max weight = %.4e\n', ...
            i, op(i, 1), op(i, 2), max(abs(weights)));
    end
    
    fprintf('Modal shape evaluation completed\n');
    
    % Save modal weights
    save('reference_data/circular_reference_modal_weights.mat', 'modal_weights', 'op', 'mode_t');

catch ME
    fprintf('Error in modal shape evaluation: %s\n', ME.message);
end

%% 3. Nonlinear Coupling Analysis
fprintf('\n3. Analyzing nonlinear coupling coefficients...\n');
try
    % Analyze H tensor properties
    H0_stats = struct();
    H0_stats.max_value = max(abs(H0(:)));
    H0_stats.mean_value = mean(abs(H0(:)));
    H0_stats.nonzero_count = sum(abs(H0(:)) > 1e-12);
    H0_stats.total_elements = numel(H0);
    H0_stats.sparsity = 1 - H0_stats.nonzero_count/H0_stats.total_elements;
    
    fprintf('H0 tensor statistics:\n');
    fprintf('- Max coefficient: %.4e\n', H0_stats.max_value);
    fprintf('- Mean coefficient: %.4e\n', H0_stats.mean_value);
    fprintf('- Non-zero elements: %d/%d (%.1f%% sparse)\n', ...
        H0_stats.nonzero_count, H0_stats.total_elements, ...
        H0_stats.sparsity*100);
    
    % Save coupling analysis
    save('reference_data/circular_reference_coupling_analysis.mat', 'H0_stats');

    fprintf('   ✅ Coupling analysis completed\n');
    
catch ME
    fprintf('   ❌ Error in coupling analysis: %s\n', ME.message);
end

%% 4. Frequency Analysis
fprintf('\n4. Computing frequency characteristics...\n');
try
    % Compute dimensional frequencies
    D = E * hd^3 / (12 * (1 - nu^2)); % Flexural rigidity
    omega_dim = sqrt(D / (rho * hd)) * (xkn / Rd^2); % Dimensional frequencies in rad/s
    freq_hz = omega_dim / (2 * pi); % Frequencies in Hz
    
    fprintf('   Frequency range: %.1f Hz to %.1f Hz\n', min(freq_hz), max(freq_hz));
    
    % Identify fundamental mode
    [~, fund_idx] = min(freq_hz);
    fprintf('   Fundamental mode: (k=%d, c=%d) at %.1f Hz\n', ...
        k_t(fund_idx), c_t(fund_idx), freq_hz(fund_idx));
    
    % Save frequency data
    save('reference_data/circular_reference_frequency_data.mat', 'omega_dim', 'freq_hz', ...
        'D', 'fund_idx');
    
    fprintf('Frequency analysis completed\n');
    
catch ME
    fprintf('Error in frequency analysis: %s\n', ME.message);
end

%% 5. Score and Excitation Setup
fprintf('\n5. Setting up excitation characteristics...\n');
% Define custom score parameters
% [StartTime, ForceMagnitude, TimeWidth, ThetaPosition (0, 2pi), RadialPosition (0,1)]
score_cell = {
    'Strike' [0.01 50.0 2e-3 1 0.5];  % Strike at center with 50N force, 2ms width
};

% Save score parameters to temporary file
CustomScoreFileName = 'Parameters/Input files/CustomScoreParCirc.mat';
save(CustomScoreFileName, 'score_cell');

try
    % Try to use score_circ if all parameters are available
    if exist('tnd', 'var') && exist('fs', 'var')
        [f_time, Tn] = score_circ(CustomScoreFileName, Rd_out, hd_out, E_out, BC_out, e, Nphi_out, scheme_out, C, k_t, c_t, xkn, ...
            JJ, II, Kkn, tnd, fs, Tsd_out);
        
        fprintf('   ✅ Excitation setup completed\n');
        fprintf('   - Force signal length: %d samples\n', length(f_time));
        fprintf('   - Modal force range: %.4e to %.4e\n', min(Tn(:)), max(Tn(:)));
    else
        fprintf('   ⚠️  Skipping excitation setup (missing time parameters)\n');
        f_time = [];
        Tn = [];
    end
    
    % Save excitation data
    save('reference_data/circular_reference_excitation_data.mat', 'f_time', 'Tn');
    
catch ME
    fprintf('Error in excitation setup: %s\n', ME.message);
    % Continue without excitation for reference data
    f_time = [];
    Tn = [];
end

%% Time simulation
switch scheme
    case 'ECS'
        [ out_nd ] = ftime_imperfect_ECS( Nphi, Npsi, Ai, H0, H1, H2, C, C1,C2, Tn, e, f_time, rp);
        
    case 'verlet'
        [ out_nd ] = ftime_imperfect_verlet( Nphi, Npsi, Ai, H1, C, C1, C2, Tn, e, f_time, rp);
    
    otherwise
        disp('Unknown scheme');
end


%% 6. Generate Comprehensive Reference Dataset
fprintf('\n6. Generating comprehensive reference dataset...\n');

reference_data = struct();

% Geometric and material properties
reference_data.geometry.Rd = Rd_out;
reference_data.geometry.hd = hd_out;
reference_data.geometry.nu = nu;
reference_data.geometry.E = E_out;
reference_data.geometry.rho = rho;
reference_data.geometry.BC = BC_out;
reference_data.geometry.Nr = Nr;
reference_data.geometry.Nth = Nth;
reference_data.geometry.dr_H = dr_H;


% Modal properties
reference_data.modal.Nphi = Nphi_out;
reference_data.modal.Npsi = Npsi_out;
reference_data.modal.k_t = k_t;
reference_data.modal.c_t = c_t;
reference_data.modal.xkn = xkn;
reference_data.modal.freq_hz = freq_hz;
reference_data.modal.mode_table = mode_t;

% Nonlinear coupling
reference_data.coupling.H0 = H0;
reference_data.coupling.H1 = H1;
reference_data.coupling.H2 = H2;
reference_data.coupling.stats = H0_stats;

% Output points and modal weights
reference_data.output.points = op;
reference_data.output.modal_weights = modal_weights;
reference_data.output.rp = rp;

% display the shape of f_time
disp(['Shape of f_time: ', num2str(size(f_time))]);
% Time simulation parameters
reference_data.simulation.scheme = scheme_out;
reference_data.simulation.fs = fs;
reference_data.simulation.fsd = fsd;
reference_data.simulation.Tsd = Tsd_out;
reference_data.simulation.tnd = tnd;
reference_data.simulation.f_time = f_time;
reference_data.simulation.Tn = Tn;
reference_data.simulation.score_cell = score_cell(2);

% Matrix coefficients
reference_data.matrices.C = C;
reference_data.matrices.C1 = C1;
reference_data.matrices.C2 = C2;
reference_data.matrices.Ai = Ai;

reference_data.output.displacement = out_nd * hd_out; % Dimensioned output
reference_data.output.velocity = diff(reference_data.output.displacement, 1, 1) * fsd; % Dimensioned output velocity

% Save complete reference dataset
save('reference_data/circular_plate_reference_complete.mat', 'reference_data');

% Export to JSON for Python compatibility
fprintf('Exporting reference data to JSON...\n');
json_str = jsonencode(reference_data);
fid = fopen('reference_data/circular_plate_reference_complete.json', 'w');
if fid == -1
    error('Cannot create JSON file');
end
fprintf(fid, '%s', json_str);
fclose(fid);
fprintf('JSON export completed: reference_data/circular_plate_reference_complete.json\n');

%% Save results
for i = 1:size(rp,1)
    audiowrite(sprintf('%s-Op_%d.wav',filename, i),reference_data.output.velocity(:,i)/1.1/max(abs(reference_data.output.velocity(:,i))),fsd);
end

%% Summary Report

fprintf('\nPlate configuration:\n');
fprintf('  - Radius: %.3f m\n', Rd_out);
fprintf('  - Thickness: %.4f m\n', hd_out);
fprintf('  - Boundary: %s\n', BC_out);
fprintf('  - Modes: %d transverse, %d in-plane\n', Nphi_out, Npsi_out);
fprintf('  - Frequency range: %.1f - %.1f Hz\n', min(freq_hz), max(freq_hz));

fprintf('\n✅ Circular plate reference data generation completed successfully!\n');
