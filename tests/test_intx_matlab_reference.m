% Fixed MATLAB script to generate reference values for intX functions
% This script ensures proper ordering between test cases and results

% Add the MATLAB functions to path
addpath('third_party/VKGong/matlab/VK-Gong/Code/Nonlinear characteristics functions/Rectangular/');

% Test parameters
L_values = [0.5, 1.0, 1.5, 2.0];
m_values = [0, 1, 2, 3, 4, 5];
p_values = [0, 1, 2, 3, 4, 5];

% Calculate total number of test cases
num_cases = length(L_values) * length(m_values) * length(p_values);

% Initialize arrays
test_cases = zeros(num_cases, 3);
int1_results = zeros(num_cases, 1);
int2_results = zeros(num_cases, 1);
int4_results = zeros(num_cases, 1);

fprintf('Generating reference data for %d test cases...\n', num_cases);

% Generate test cases and results in proper order
case_idx = 1;
for i = 1:length(L_values)
    L = L_values(i);
    for j = 1:length(m_values)
        m = m_values(j);
        for k = 1:length(p_values)
            p = p_values(k);
            
            % Store test case
            test_cases(case_idx, :) = [L, m, p];
            
            % Calculate results
            try
                int1_results(case_idx) = int1(m, p, L);
            catch ME
                fprintf('Error in int1(%d, %d, %.1f): %s\n', m, p, L, ME.message);
                int1_results(case_idx) = NaN;
            end
            
            try
                int2_results(case_idx) = int2(m, p, L);
            catch ME
                fprintf('Error in int2(%d, %d, %.1f): %s\n', m, p, L, ME.message);
                int2_results(case_idx) = NaN;
            end
            
            try
                int4_results(case_idx) = int4(m, p, L);
            catch ME
                fprintf('Error in int4(%d, %d, %.1f): %s\n', m, p, L, ME.message);
                int4_results(case_idx) = NaN;
            end
            
            % Debug output for first few cases
            if case_idx <= 10
                fprintf('Case %d: L=%.1f, m=%d, p=%d -> int1=%.6e, int2=%.6e, int4=%.6e\n', ...
                    case_idx, L, m, p, int1_results(case_idx), int2_results(case_idx), int4_results(case_idx));
            end
            
            case_idx = case_idx + 1;
        end
    end
end

% Create JSON structure
json_results = struct();
json_results.L_values = L_values;
json_results.m_values = m_values;
json_results.p_values = p_values;
json_results.test_cases = test_cases;
json_results.int1_results = int1_results;
json_results.int2_results = int2_results;
json_results.int4_results = int4_results;

% Create reference_data directory if it doesn't exist
if ~exist('reference_data', 'dir')
    mkdir('reference_data');
end

% Save as MAT file
save('reference_data/test_intx_matlab_reference_results.mat', 'json_results');
fprintf('Results saved to MAT file\n');

% Write JSON file
json_text = jsonencode(json_results);

fid = fopen('reference_data/test_intx_matlab_reference_results.json', 'w');
fprintf(fid, '%s', json_text);
fclose(fid);
fprintf('Results saved to test_intx_matlab_reference_results.json\n');

% Verify first few cases
fprintf('\nVerification of first few cases:\n');
for i = 1:5
    L = test_cases(i, 1);
    m = test_cases(i, 2);
    p = test_cases(i, 3);
    fprintf('Case %d: int1(%d,%d,%.1f) = %.6e\n', i, m, p, L, int1_results(i));
end

fprintf('\nMATLAB reference generation complete!\n');