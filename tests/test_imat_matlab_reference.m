%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% MATLAB Reference Data Generation for i*_mat functions
% Generates comprehensive test data for all i*_mat functions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add the MATLAB reference functions to path
addpath('../third_party/VKGong/matlab/VK-Gong/Code/Nonlinear characteristics functions/Rectangular');

% Define test parameter ranges
Npsi_values = [2, 3, 4, 5];  % Different matrix sizes
Nphi_values = [2, 3, 4, 5];  % Different matrix sizes  
L_values = [0.5, 1.0, 1.5, 2.0];  % Length parameter

% Calculate total number of test cases
num_cases = length(Npsi_values) * length(Nphi_values) * length(L_values);
fprintf('Generating %d test cases for i*_mat functions\n', num_cases);

% Initialize storage for test cases and results
test_cases = zeros(num_cases, 3);  % [Npsi, Nphi, L]

% Initialize result storage for each function
% Note: Results will be flattened 3D arrays since JSON doesn't handle 3D well
i1_results = cell(num_cases, 1);
i2_results = cell(num_cases, 1);
i3_results = cell(num_cases, 1);
i4_results = cell(num_cases, 1);
i5_results = cell(num_cases, 1);
i9_results = cell(num_cases, 1);
i10_results = cell(num_cases, 1);
i11_results = cell(num_cases, 1);
i12_results = cell(num_cases, 1);
i13_results = cell(num_cases, 1);

% Generate test cases and results
case_idx = 1;
for i = 1:length(Npsi_values)
    Npsi = Npsi_values(i);
    for j = 1:length(Nphi_values)
        Nphi = Nphi_values(j);
        for k = 1:length(L_values)
            L = L_values(k);
            
            % Store test case parameters
            test_cases(case_idx, :) = [Npsi, Nphi, L];
            
            fprintf('Computing case %d/%d: Npsi=%d, Nphi=%d, L=%.1f\n', ...
                case_idx, num_cases, Npsi, Nphi, L);
            
            % Compute results for all i*_mat functions
            % Store as flattened arrays (will reshape in Python)
            try
                result = i1_mat(Npsi, Nphi, L);
                i1_results{case_idx} = result(:)';  % Flatten to row vector
            catch ME
                fprintf('Error in i1_mat: %s\n', ME.message);
                i1_results{case_idx} = NaN;
            end
            
            try
                result = i2_mat(Npsi, Nphi, L);
                i2_results{case_idx} = result(:)';
            catch ME
                fprintf('Error in i2_mat: %s\n', ME.message);
                i2_results{case_idx} = NaN;
            end
            
            try
                result = i3_mat(Npsi, Nphi, L);
                i3_results{case_idx} = result(:)';
            catch ME
                fprintf('Error in i3_mat: %s\n', ME.message);
                i3_results{case_idx} = NaN;
            end
            
            try
                result = i4_mat(Npsi, Nphi, L);
                i4_results{case_idx} = result(:)';
            catch ME
                fprintf('Error in i4_mat: %s\n', ME.message);
                i4_results{case_idx} = NaN;
            end
            
            try
                result = i5_mat(Npsi, Nphi, L);
                i5_results{case_idx} = result(:)';
            catch ME
                fprintf('Error in i5_mat: %s\n', ME.message);
                i5_results{case_idx} = NaN;
            end
            
            try
                result = i9_mat(Npsi, Nphi, L);
                i9_results{case_idx} = result(:)';
            catch ME
                fprintf('Error in i9_mat: %s\n', ME.message);
                i9_results{case_idx} = NaN;
            end
            
            try
                result = i10_mat(Npsi, Nphi, L);
                i10_results{case_idx} = result(:)';
            catch ME
                fprintf('Error in i10_mat: %s\n', ME.message);
                i10_results{case_idx} = NaN;
            end
            
            try
                result = i11_mat(Npsi, Nphi, L);
                i11_results{case_idx} = result(:)';
            catch ME
                fprintf('Error in i11_mat: %s\n', ME.message);
                i11_results{case_idx} = NaN;
            end
            
            try
                result = i12_mat(Npsi, Nphi, L);
                i12_results{case_idx} = result(:)';
            catch ME
                fprintf('Error in i12_mat: %s\n', ME.message);
                i12_results{case_idx} = NaN;
            end
            
            try
                result = i13_mat(Npsi, Nphi, L);
                i13_results{case_idx} = result(:)';
            catch ME
                fprintf('Error in i13_mat: %s\n', ME.message);
                i13_results{case_idx} = NaN;
            end
            
            case_idx = case_idx + 1;
        end
    end
end

% Create structure for JSON export
data = struct();
data.test_cases = test_cases;
data.i1_results = i1_results;
data.i2_results = i2_results;
data.i3_results = i3_results;
data.i4_results = i4_results;
data.i5_results = i5_results;
data.i9_results = i9_results;
data.i10_results = i10_results;
data.i11_results = i11_results;
data.i12_results = i12_results;
data.i13_results = i13_results;

% Export to JSON
% Create reference_data directory if it doesn't exist
if ~exist('reference_data', 'dir')
    mkdir('reference_data');
end

json_filename = 'reference_data/test_imat_matlab_reference_results.json';
fprintf('Exporting results to %s\n', json_filename);

% Convert to JSON string
json_str = jsonencode(data);

% Write to file
fid = fopen(json_filename, 'w');
if fid == -1
    error('Could not open file for writing: %s', json_filename);
end
fprintf(fid, '%s', json_str);
fclose(fid);

fprintf('Successfully generated reference data for i*_mat functions\n');
fprintf('Test cases: %d\n', num_cases);
fprintf('Functions tested: i1_mat, i2_mat, i3_mat, i4_mat, i5_mat, i9_mat, i10_mat, i11_mat, i12_mat, i13_mat\n');