% MATLAB script to generate reference values for H_tensor_rectangular function
% This script ensures proper ordering between test cases and results following the established pattern

% Add the MATLAB functions to path (correct relative path from tests directory)
addpath('../third_party/VKGong/matlab/VK-Gong/Code/Nonlinear characteristics functions/Rectangular/');

% Test parameters following the pattern from AiryStressFactorsCalculation.m
Npsi_values = [5, 10];
Nphi_values = [3, 5];
Lx_values = [1.0, 1.5];
Ly_values = [1.0, 1.5];

% Mode indices will be generated automatically by ComputeTransverseEigenfrequenciesRectangular

% Calculate total number of test cases
num_cases = length(Npsi_values) * length(Nphi_values) * length(Lx_values) * length(Ly_values);

% Initialize arrays with explicit case indexing
test_cases = zeros(num_cases, 4); % [Npsi, Nphi, Lx, Ly]
kx_indices_results = cell(num_cases, 1);
ky_indices_results = cell(num_cases, 1);

% Initialize arrays for coefficient inputs (from MATLAB eigendecomposition)
matlab_coeff0_inputs = cell(num_cases, 1);
matlab_coeff1_inputs = cell(num_cases, 1);
matlab_coeff2_inputs = cell(num_cases, 1);

% Initialize arrays for H tensor results
H0_results = cell(num_cases, 1);
H1_results = cell(num_cases, 1);
H2_results = cell(num_cases, 1);

fprintf('Generating H_tensor_rectangular reference data for %d test cases...\n', num_cases);

% Generate test cases and results in proper order with explicit case indexing
case_idx = 1;
for i = 1:length(Npsi_values)
    Npsi = Npsi_values(i);
    for j = 1:length(Nphi_values)
        Nphi = Nphi_values(j);
        
        % Mode indices will be computed by ComputeTransverseEigenfrequenciesRectangular
        
        for k = 1:length(Lx_values)
            Lx = Lx_values(k);
            for l = 1:length(Ly_values)
                Ly = Ly_values(l);
                
                % Store test case
                test_cases(case_idx, :) = [Npsi, Nphi, Lx, Ly];
                
                fprintf('Computing case %d/%d: Npsi=%d, Nphi=%d, Lx=%.1f, Ly=%.1f\n', ...
                    case_idx, num_cases, Npsi, Nphi, Lx, Ly);
                
                try
                    % 1. Use AiryStressFactorsCalculation.m to compute coefficients
                    BC = 'SimplySupported';
                    [coeff0_final, coeff1_final, coeff2_final] = AiryStressFactorsCalculation(BC, Npsi, Lx, Ly);
                    
                    % Store coefficient inputs for reproducibility
                    matlab_coeff0_inputs{case_idx} = coeff0_final(:)';
                    matlab_coeff1_inputs{case_idx} = coeff1_final(:)';
                    matlab_coeff2_inputs{case_idx} = coeff2_final(:)';
                    
                    % 2. Use ComputeTransverseEigenfrequenciesRectangular to get proper mode_t structure
                    [mode_t] = ComputeTransverseEigenfrequenciesRectangular(BC, Lx, Ly, Nphi);
                    
                    % Extract kx and ky indices from mode_t structure for storage
                    kx_indices = mode_t(:, 2)';  % x-direction indices
                    ky_indices = mode_t(:, 3)';  % y-direction indices
                    kx_indices_results{case_idx} = kx_indices;
                    ky_indices_results{case_idx} = ky_indices;
                    
                    % 3. Now call H_tensorRectangular with these coefficients
                    [H0, H1, H2] = H_tensorRectangular(coeff0_final, coeff1_final, coeff2_final, ...
                                                       Nphi, Npsi, Lx, Ly, mode_t, BC);
                    
                    % Store H tensor results (flatten matrices in column-major order for JSON)
                    H0_results{case_idx} = H0(:)';
                    H1_results{case_idx} = H1(:)';
                    H2_results{case_idx} = H2(:)';
                    
                    % Debug output for first few cases
                    if case_idx <= 3
                        fprintf('Case %d: coeff size=%dx%d, H0 size=%dx%d\n', ...
                            case_idx, size(coeff0_final,1), size(coeff0_final,2), ...
                            size(H0,1), size(H0,2));
                    end

                catch ME
                    fprintf('Error in case %d: %s\n', case_idx, ME.message);
                    matlab_coeff0_inputs{case_idx} = NaN;
                    matlab_coeff1_inputs{case_idx} = NaN;
                    matlab_coeff2_inputs{case_idx} = NaN;
                    H0_results{case_idx} = NaN;
                    H1_results{case_idx} = NaN;
                    H2_results{case_idx} = NaN;
                end
                
                case_idx = case_idx + 1;
            end
        end
    end
end

% Create JSON structure
json_results = struct();
json_results.Npsi_values = Npsi_values;
json_results.Nphi_values = Nphi_values;
json_results.Lx_values = Lx_values;
json_results.Ly_values = Ly_values;
json_results.test_cases = test_cases;
json_results.kx_indices_results = kx_indices_results;
json_results.ky_indices_results = ky_indices_results;

% Add coefficient inputs (for reproducibility in Python tests)
json_results.matlab_coeff0_inputs = matlab_coeff0_inputs;
json_results.matlab_coeff1_inputs = matlab_coeff1_inputs;
json_results.matlab_coeff2_inputs = matlab_coeff2_inputs;

% Add H tensor results
json_results.H0_results = H0_results;
json_results.H1_results = H1_results;
json_results.H2_results = H2_results;

% Save as MAT file
save('reference_data/test_H_tensor_matlab_reference_results.mat', 'json_results');
fprintf('Results saved to MAT file\n');

% Write JSON file
json_text = jsonencode(json_results);
% Create reference_data directory if it doesn't exist
if ~exist('reference_data', 'dir')
    mkdir('reference_data');
end

json_filename = 'reference_data/test_H_tensor_matlab_reference_results.json';
fid = fopen(json_filename, 'w');
if fid == -1
    error('Could not open file for writing: %s', json_filename);
end
fprintf(fid, '%s', json_text);
fclose(fid);
fprintf('Results saved to test_H_tensor_matlab_reference_results.json\n');

% Verify first few cases
fprintf('\nVerification of first few cases:\n');
for i = 1:min(3, num_cases)
    Npsi = test_cases(i, 1);
    Nphi = test_cases(i, 2);
    Lx = test_cases(i, 3);
    Ly = test_cases(i, 4);
    fprintf('Case %d: Npsi=%d, Nphi=%d, Lx=%.1f, Ly=%.1f, kx=%s, ky=%s\n', ...
        i, Npsi, Nphi, Lx, Ly, mat2str(kx_indices_results{i}), mat2str(ky_indices_results{i}));
end

fprintf('\nH_tensor_rectangular MATLAB reference generation complete!\n');