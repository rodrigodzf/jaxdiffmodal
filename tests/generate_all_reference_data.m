function generate_all_reference_data()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                   Generate All Reference Data                     %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script generates all reference data files used by the Python test suite.
% It runs all the individual reference generation scripts.
%
% Usage:
%   cd tests
%   matlab -batch "generate_all_reference_data"
%
% Output:
%   All reference data files are saved to tests/reference_data/
%

fprintf('=================================================\n');
fprintf('Generating All Reference Data for Test Suite\n');
fprintf('=================================================\n\n');

% Create reference_data directory if it doesn't exist
if ~exist('reference_data', 'dir')
    mkdir('reference_data');
    fprintf('Created reference_data/ directory\n\n');
end

% Add necessary paths
addpath(genpath('../third_party/VKGong/matlab/VK-Gong/Code'));
addpath('patched');  % For patched functions

run('test_intx_matlab_reference.m');
run('test_K_M_matlab_reference.m');
run('test_H_tensor_matlab_reference.m');
run('test_circular_matlab_reference.m');
run('test_evaluate_circular_modes_matlab_reference.m');
run('test_imat_matlab_reference.m');
% run('circular_membrane_reference_simple.m');
run('mainRectangularCustom_reference.m');
run('mainCircularCustom_reference.m');

fprintf('\n=================================================\n');
fprintf('Reference Data Generation Complete\n');
fprintf('=================================================\n');
fprintf('All reference data files are now in tests/reference_data/\n');
fprintf('Run Python tests to verify everything works correctly.\n');

end