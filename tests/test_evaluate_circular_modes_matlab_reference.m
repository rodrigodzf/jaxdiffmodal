function test_evaluate_circular_modes_matlab_reference()
% TEST_EVALUATE_CIRCULAR_MODES_MATLAB_REFERENCE Generate reference data
% for testing evaluate_circular_modes function against MATLAB implementation
%
% This script generates comprehensive test data using the MATLAB reference
% implementation and exports results to JSON for Python consumption.

fprintf('Generating MATLAB reference data for evaluate_circular_modes...\n');

% Add path to VKGong functions and patched directory
addpath(genpath('../third_party/VKGong/matlab/VK-Gong/Code'));
addpath('patched');

%% Test Parameters
R_values = [1.0, 2.0];                    % Plate radius
dr_values = [0.01, 0.02];                 % Integration step
nu_values = [0.3, 0.25];                  % Poisson ratio
BC_values = {'free', 'clamped'};          % Boundary conditions

% Test points on the plate [theta, r]
test_points = [
    0.0, 0.5;      % theta=0, r=0.5 (center line)
    pi/4, 0.7;     % theta=45deg, r=0.7
    pi/2, 0.9;     % theta=90deg, r=0.9 (near edge)
    pi, 0.3;       % theta=180deg, r=0.3
    3*pi/2, 0.8    % theta=270deg, r=0.8
];

% Mode tables for testing (realistic eigenvalues)
mode_t_free = [
    1, 2.404, 0, 1, 1, 5.783;     % (0,1) cos mode
    2, 3.832, 1, 1, 1, 14.684;    % (1,1) cos mode  
    3, 3.832, 1, 1, 2, 14.684;    % (1,1) sin mode
    4, 5.135, 2, 1, 1, 26.368;    % (2,1) cos mode
    5, 5.135, 2, 1, 2, 26.368     % (2,1) sin mode
];

mode_t_clamped = [
    1, 3.196, 0, 1, 1, 10.214;    % (0,1) cos mode
    2, 4.611, 1, 1, 1, 21.261;    % (1,1) cos mode
    3, 4.611, 1, 1, 2, 21.261;    % (1,1) sin mode  
    4, 5.906, 2, 1, 1, 34.881;    % (2,1) cos mode
    5, 5.906, 2, 1, 2, 34.881     % (2,1) sin mode
];

%% Generate test cases
test_cases = [];
results = [];
case_idx = 1;

% Systematic parameter sweep
for bc_idx = 1:length(BC_values)
    BC = BC_values{bc_idx};
    
    % Select appropriate mode table and KR value
    if strcmp(BC, 'free')
        mode_t = mode_t_free;
        KR = 0;  % Free boundary
    else  % clamped
        mode_t = mode_t_clamped;
        KR = inf;  % Clamped boundary
    end
    
    for R_idx = 1:length(R_values)
        R = R_values(R_idx);
        for dr_idx = 1:length(dr_values)
            dr = dr_values(dr_idx);
            for nu_idx = 1:length(nu_values)
                nu = nu_values(nu_idx);
                for pt_idx = 1:size(test_points, 1)
                    op = test_points(pt_idx, :);
                    
                    % Store test parameters
                    test_cases(case_idx, :) = [bc_idx, R_idx, dr_idx, nu_idx, pt_idx];
                    
                    % Call MATLAB reference function
                    try
                        weights = evaluate_circular_modes(mode_t, nu, KR, op, BC, 1, dr / R);
                        results{case_idx} = weights(:)';  % Store as row vector
                        
                        fprintf('Case %d: BC=%s, R=%.1f, dr=%.2f, nu=%.2f, theta=%.3f, r=%.3f - Success\n', ...
                               case_idx, BC, R, dr, nu, op(1), op(2));
                    catch ME
                        fprintf('Case %d: BC=%s, R=%.1f, dr=%.2f, nu=%.2f, theta=%.3f, r=%.3f - Error: %s\n', ...
                               case_idx, BC, R, dr, nu, op(1), op(2), ME.message);
                        results{case_idx} = [];
                    end
                    
                    case_idx = case_idx + 1;
                end
            end
        end
    end
end

%% Prepare data structure for JSON export
data_struct = struct();

% Test parameters
data_struct.R_values = R_values;
data_struct.dr_values = dr_values;
data_struct.nu_values = nu_values;
data_struct.BC_values = {BC_values};  % Cell array for JSON
data_struct.test_points = test_points;

% Mode tables
data_struct.mode_t_free = mode_t_free;
data_struct.mode_t_clamped = mode_t_clamped;

% Test cases and results
data_struct.test_cases = test_cases;
data_struct.num_cases = size(test_cases, 1);

% Results (handle empty results for failed cases)
data_struct.results = cell(1, length(results));
for i = 1:length(results)
    if ~isempty(results{i})
        data_struct.results{i} = results{i};
    else
        data_struct.results{i} = [];
    end
end

%% Export to JSON
% Create reference_data directory if it doesn't exist
if ~exist('reference_data', 'dir')
    mkdir('reference_data');
end

json_filename = 'reference_data/test_evaluate_circular_modes_matlab_reference_results.json';
fprintf('Exporting results to %s...\n', json_filename);

try
    json_str = jsonencode(data_struct);
    fid = fopen(json_filename, 'w');
    if fid == -1
        error('Could not open file for writing: %s', json_filename);
    end
    fprintf(fid, '%s', json_str);
    fclose(fid);
    fprintf('✅ Successfully exported %d test cases to JSON\n', data_struct.num_cases);
catch ME
    fprintf('❌ Error exporting to JSON: %s\n', ME.message);
    
    % Fallback: save as .mat file
    mat_filename = 'test_evaluate_circular_modes_matlab_reference_results.mat';
    fprintf('Saving fallback .mat file: %s\n', mat_filename);
    save(mat_filename, 'data_struct', '-v7');
end

fprintf('MATLAB reference data generation completed.\n');

end