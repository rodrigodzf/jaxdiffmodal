%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                                                                   %
%                              VK-Gong                              % 
%                                                                   %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This script generates reference data for testing Python circular plate 
% functions against MATLAB implementation.
% 
% It tests:
% - CosCosCosIntegration and CosSinSinIntegration functions
% - HcoeffCircular function with Bessel function calculations
% - H_tensorCircular function for nonlinear coupling tensors
% - ComputeTransverseEigenfrequenciesCircular and ComputeInPlaneEigenfrequenciesCircular
%
% The results are exported to JSON format for consumption by Python tests.

clear all;
clc;

% Add paths to VK-Gong functions
addpath('patched');
addpath('../third_party/VKGong/matlab/VK-Gong/Code/Nonlinear characteristics functions/Circular');
addpath('../third_party/VKGong/matlab/VK-Gong/Code/Linear characteristics functions/Circular');

%% Test 1: Angular Integration Functions

% Test parameters for CosCosCosIntegration
disp('Testing CosCosCosIntegration...');
cos_cos_cos_test_cases = [];
cos_cos_cos_results = [];

k_values = [0, 1, 2, 3];
l_values = [0, 1, 2, 3];
m_values = [0, 1, 2, 3];

case_idx = 1;
for i = 1:length(k_values)
    k = k_values(i);
    for j = 1:length(l_values)
        l = l_values(j);
        for m_idx = 1:length(m_values)
            m = m_values(m_idx);
            
            cos_cos_cos_test_cases(case_idx, :) = [k, l, m];
            cos_cos_cos_results(case_idx) = CosCosCosIntegration(k, l, m);
            case_idx = case_idx + 1;
        end
    end
end

% Test parameters for CosSinSinIntegration
disp('Testing CosSinSinIntegration...');
cos_sin_sin_test_cases = [];
cos_sin_sin_results = [];

case_idx = 1;
for i = 1:length(k_values)
    k = k_values(i);
    for j = 1:length(l_values)
        l = l_values(j);
        for m_idx = 1:length(m_values)
            m = m_values(m_idx);
            
            cos_sin_sin_test_cases(case_idx, :) = [k, l, m];
            cos_sin_sin_results(case_idx) = CosSinSinIntegration(k, l, m);
            case_idx = case_idx + 1;
        end
    end
end

%% Test 2: H-coefficient Computation

disp('Testing HcoeffCircular...');
hcoeff_test_cases = [];
hcoeff_results = [];

% Test parameters for HcoeffCircular
kp_vals = [1, 2];
kq_vals = [1, 2];
cp_vals = [1, 2];
cq_vals = [1, 2];
xip_vals = [1.0, 2.0];
xiq_vals = [1.0, 2.0];
ki_vals = [1, 2];
ci_vals = [1, 2];
zeta_vals = [1.0, 2.0];
nu_vals = [0.3];
KR_vals = [0.0];
dr_H_vals = [0.01];

case_idx = 1;
for i1 = 1:length(kp_vals)
    for i2 = 1:length(kq_vals)
        for i3 = 1:length(cp_vals)
            for i4 = 1:length(cq_vals)
                for i5 = 1:length(xip_vals)
                    for i6 = 1:length(xiq_vals)
                        for i7 = 1:length(ki_vals)
                            for i8 = 1:length(ci_vals)
                                for i9 = 1:length(zeta_vals)
                                    for i10 = 1:length(nu_vals)
                                        for i11 = 1:length(KR_vals)
                                            for i12 = 1:length(dr_H_vals)
                                                kp = kp_vals(i1);
                                                kq = kq_vals(i2);
                                                cp = cp_vals(i3);
                                                cq = cq_vals(i4);
                                                xip = xip_vals(i5);
                                                xiq = xiq_vals(i6);
                                                ki = ki_vals(i7);
                                                ci = ci_vals(i8);
                                                zeta = zeta_vals(i9);
                                                nu = nu_vals(i10);
                                                KR = KR_vals(i11);
                                                dr_H = dr_H_vals(i12);
                                                
                                                hcoeff_test_cases(case_idx, :) = [kp, kq, cp, cq, xip, xiq, ki, ci, zeta, nu, KR, dr_H];
                                                hcoeff_results(case_idx) = HcoefficientCircular(kp, kq, cp, cq, xip, xiq, ki, ci, zeta, nu, KR, dr_H);
                                                case_idx = case_idx + 1;
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

%% Test 3: H-tensor Computation with Mode Generation

disp('Testing H_tensorCircular_modified with mode generation...');

% Add path to our modified H tensor function
addpath('..');

% Test configurations for H tensor computation
H_tensor_configs = {
    % [Nphi, Npsi, BC, nu, KR, KT, dr_H]
    {4, 4, 'free', 0.3, 0.0, 0.0, 0.01},
    {6, 6, 'free', 0.3, 0.0, 0.0, 0.01},
    {4, 4, 'clamped', 0.3, inf, inf, 0.01},
    {6, 6, 'clamped', 0.3, inf, inf, 0.01},
    {4, 4, 'elastic', 0.3, 1.0, 1.0, 0.01},
    {6, 6, 'elastic', 0.3, 2.0, 2.0, 0.01}
};

H_tensor_test_cases = {};
H_tensor_H0_results = {};
H_tensor_H1_results = {};
H_tensor_H2_results = {};

for config_idx = 1:length(H_tensor_configs)
    config = H_tensor_configs{config_idx};
    Nphi = config{1};
    Npsi = config{2}; 
    BC = config{3};
    nu = config{4};
    KR = config{5};
    KT = config{6};
    dr_H = config{7};
    
    disp(sprintf('Computing H tensor for config %d: Nphi=%d, Npsi=%d, BC=%s', ...
                 config_idx, Nphi, Npsi, BC));
    
    % Generate modes using eigenfrequency functions
    dx = 0.01;
    xmax = 20.0;
    
    try
        % Compute transverse modes
        [mode_t, ~] = ComputeTransverseEigenfrequenciesCircular(dx, xmax, BC, nu, KR, KT);
        
        % Compute in-plane modes  
        [mode_l, ~] = ComputeInPlaneEigenfrequenciesCircular(dx, xmax, BC, nu);
        
        % Check if we have enough modes
        if size(mode_t, 1) < Nphi
            disp(sprintf('Warning: Only %d transverse modes found, need %d', size(mode_t, 1), Nphi));
            continue;
        end
        if size(mode_l, 1) < Npsi
            disp(sprintf('Warning: Only %d in-plane modes found, need %d', size(mode_l, 1), Npsi));
            continue;
        end
        
        % Store test case parameters (truncate modes to required size)
        H_tensor_test_cases{config_idx} = struct(...
            'Nphi', Nphi, 'Npsi', Npsi, 'BC', BC, 'nu', nu, ...
            'KR', KR, 'KT', KT, 'dr_H', dr_H, ...
            'mode_t', mode_t(1:Nphi, :), 'mode_l', mode_l(1:Npsi, :));
        
        % Compute H tensors using modified MATLAB function with computed modes
        [H0, H1, H2] = H_tensorCircular_modified(Nphi, Npsi, BC, nu, KR, KT, dr_H, mode_t, mode_l);
        
        % Store results (flatten matrices in column-major order for JSON export)
        H_tensor_H0_results{config_idx} = H0(:)';
        H_tensor_H1_results{config_idx} = H1(:)';
        H_tensor_H2_results{config_idx} = H2(:)';
        
        disp(sprintf('Successfully computed H tensor for config %d', config_idx));
        
    catch ME
        disp(sprintf('Error computing H tensor for config %d: %s', config_idx, ME.message));
        % Store empty results for failed cases
        H_tensor_test_cases{config_idx} = struct();
        H_tensor_H0_results{config_idx} = [];
        H_tensor_H1_results{config_idx} = [];
        H_tensor_H2_results{config_idx} = [];
    end
end

%% Test 4: Circular Plate Eigenvalues

disp('Testing ComputeTransverseEigenfrequenciesCircular...');
transverse_eig_test_cases = {};
transverse_eig_results = {};

% Test parameters for transverse eigenvalues
dx_vals = [0.01];
xmax_vals = [20.0];
BC_vals = {'free', 'clamped'};
nu_vals = [0.3];
KR_vals = [0.0];
KT_vals = [0.0];

case_idx = 1;
for i1 = 1:length(dx_vals)
    for i2 = 1:length(xmax_vals)
        for i3 = 1:length(BC_vals)
            for i4 = 1:length(nu_vals)
                for i5 = 1:length(KR_vals)
                    for i6 = 1:length(KT_vals)
                        dx = dx_vals(i1);
                        xmax = xmax_vals(i2);
                        BC = BC_vals{i3};
                        nu = nu_vals(i4);
                        KR = KR_vals(i5);
                        KT = KT_vals(i6);
                        
                        transverse_eig_test_cases{case_idx} = {dx, xmax, BC, nu, KR, KT};
                        
                        % Compute eigenvalues
                        [mode_t, Zeros] = ComputeTransverseEigenfrequenciesCircular(dx, xmax, BC, nu, KR, KT);
                        
                        % Store results
                        transverse_eig_results{case_idx} = struct('mode_t', mode_t, 'zeros', Zeros);
                        
                        case_idx = case_idx + 1;
                    end
                end
            end
        end
    end
end

disp('Testing ComputeInPlaneEigenfrequenciesCircular...');
inplane_eig_test_cases = {};
inplane_eig_results = {};

% Test parameters for in-plane eigenvalues
case_idx = 1;
for i1 = 1:length(dx_vals)
    for i2 = 1:length(xmax_vals)
        for i3 = 1:length(BC_vals)
            for i4 = 1:length(nu_vals)
                dx = dx_vals(i1);
                xmax = xmax_vals(i2);
                BC = BC_vals{i3};
                nu = nu_vals(i4);
                
                inplane_eig_test_cases{case_idx} = {dx, xmax, BC, nu};
                
                % Compute eigenvalues
                [mode_l, Zeros] = ComputeInPlaneEigenfrequenciesCircular(dx, xmax, BC, nu);
                
                % Store results
                inplane_eig_results{case_idx} = struct('mode_l', mode_l, 'zeros', Zeros);
                
                case_idx = case_idx + 1;
            end
        end
    end
end

%% Export Results to JSON

disp('Exporting results to JSON...');

% Create results structure
results = struct();
results.cos_cos_cos_test_cases = cos_cos_cos_test_cases;
results.cos_cos_cos_results = cos_cos_cos_results;
results.cos_sin_sin_test_cases = cos_sin_sin_test_cases;
results.cos_sin_sin_results = cos_sin_sin_results;
results.hcoeff_test_cases = hcoeff_test_cases;
results.hcoeff_results = hcoeff_results;
results.H_tensor_test_cases = H_tensor_test_cases;
results.H_tensor_H0_results = H_tensor_H0_results;
results.H_tensor_H1_results = H_tensor_H1_results;
results.H_tensor_H2_results = H_tensor_H2_results;
results.transverse_eig_test_cases = transverse_eig_test_cases;
results.transverse_eig_results = transverse_eig_results;
results.inplane_eig_test_cases = inplane_eig_test_cases;
results.inplane_eig_results = inplane_eig_results;

% Convert to JSON format
json_str = jsonencode(results);

% Write to file
% Create reference_data directory if it doesn't exist
if ~exist('reference_data', 'dir')
    mkdir('reference_data');
end

json_filename = 'reference_data/test_circular_matlab_reference_results.json';
fid = fopen(json_filename, 'w');
if fid == -1
    error('Cannot create JSON file: %s', json_filename);
end
fprintf(fid, '%s', json_str);
fclose(fid);

% Also save as .mat file for debugging
mat_filename = 'reference_data/test_circular_matlab_reference_results.mat';
save(mat_filename, 'results');

disp(['Results saved to: ' json_filename]);
disp(['MATLAB data saved to: ' mat_filename]);
disp('Test data generation complete!');