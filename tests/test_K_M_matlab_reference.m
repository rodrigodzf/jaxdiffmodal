% Fixed MATLAB script to generate reference values for K, M matrices and eigendecomposition
% This script ensures proper ordering between test cases and results following the established pattern

% Add the MATLAB functions to path (correct relative path from tests directory)
addpath('../third_party/VKGong/matlab/VK-Gong/Code/Nonlinear characteristics functions/Rectangular/');

% Test parameters following the pattern from AiryStressFactorsCalculation.m
Npsi_values = [5, 10];
Lx_values = [1.0, 1.5];
Ly_values = [1.0, 1.5];

% Calculate total number of test cases
num_cases = length(Npsi_values) * length(Lx_values) * length(Ly_values);

% Initialize arrays with explicit case indexing
test_cases = zeros(num_cases, 3); % [Npsi, Lx, Ly]
K_results = cell(num_cases, 1);
M_results = cell(num_cases, 1);
eig_vals_results = cell(num_cases, 1);
eig_vecs_results = cell(num_cases, 1);

% Initialize arrays for airy_stress_coefficients results
airy_coeff0_results = cell(num_cases, 1);
airy_coeff1_results = cell(num_cases, 1);
airy_coeff2_results = cell(num_cases, 1);
airy_auto_results = cell(num_cases, 1);
airy_S_results = cell(num_cases, 1);

fprintf('Generating reference data for %d test cases...\n', num_cases);

% Generate test cases and results in proper order with explicit case indexing
case_idx = 1;
for i = 1:length(Npsi_values)
    Npsi = Npsi_values(i);
    for j = 1:length(Lx_values)
        Lx = Lx_values(j);
        for k = 1:length(Ly_values)
            Ly = Ly_values(k);
            
            % Store test case
            test_cases(case_idx, :) = [Npsi, Lx, Ly];
            
            fprintf('Computing case %d/%d: Npsi=%d, Lx=%.1f, Ly=%.1f\n', ...
                case_idx, num_cases, Npsi, Lx, Ly);
            
            try
                % 1. Precompute integral matrices following AiryStressFactorsCalculation pattern
                I1x = zeros(Npsi, Npsi);
                I2x = zeros(Npsi, Npsi);
                I4x = zeros(Npsi, Npsi);
                I1y = zeros(Npsi, Npsi);
                I2y = zeros(Npsi, Npsi);
                I4y = zeros(Npsi, Npsi);

                for m_idx = 1:Npsi
                    for p_idx = 1:Npsi
                        m = m_idx - 1;  % Convert to 0-based indexing
                        p = p_idx - 1;  % Convert to 0-based indexing
                        
                        I1x(m_idx, p_idx) = int1(m, p, Lx);
                        I2x(m_idx, p_idx) = int2(m, p, Lx);
                        I4x(m_idx, p_idx) = int4(m, p, Lx);

                        I1y(m_idx, p_idx) = int1(m, p, Ly);
                        I2y(m_idx, p_idx) = int2(m, p, Ly);
                        I4y(m_idx, p_idx) = int4(m, p, Ly);
                    end
                end
                
                % 2. Assemble K and M matrices following AiryStressFactorsCalculation pattern
                K = kron(I1x, I2y) + kron(I2x, I1y) + 2.0 * kron(I4x, I4y);
                M = kron(I2x, I2y);
                
                % Store results (flatten matrices in column-major order for JSON)
                K_results{case_idx} = K(:)';
                M_results{case_idx} = M(:)';
                
                % 3. Solve eigenvalue problem
                [VEC, VAL] = eig(K, M);
                vals = diag(VAL);
                
                % 4. Sort eigenvalues and eigenvectors for consistent comparison
                [sorted_vals, sort_idx] = sort(real(vals));
                
                % Handle potential complex values properly for JSON
                eig_vals_results{case_idx} = struct('real', real(sorted_vals), 'imag', imag(sorted_vals));
                
                sorted_vecs = VEC(:, sort_idx);
                eig_vecs_results{case_idx} = struct('real', real(sorted_vecs(:)'), 'imag', imag(sorted_vecs(:)'));
                
                % 5. Process eigenvalues and eigenvectors following AiryStressFactorsCalculation.m lines 65+
                
                % Find indices of negative or imaginary eigenvalues and remove them
                [index_of_neg] = find(diag(VAL) < 0);
                [index_of_imag] = find(imag(diag(VAL)) ~= 0);  % More explicit check for imaginary parts
                VAL_clean = VAL;
                VEC_clean = VEC;
                VAL_clean(:, index_of_neg) = [];
                VEC_clean(:, index_of_neg) = [];
                VAL_clean(:, index_of_imag) = [];
                VEC_clean(:, index_of_imag) = [];
                
                % Sort the cleaned eigenvalues and eigenvectors
                if ~isempty(VAL_clean)
                    tres = find(VAL_clean);
                    [auto, v] = sort(VAL_clean(tres));
                    S = length(auto);
                    
                    coeff = VEC_clean(:, v);
                    
                    % Initialize coefficient arrays
                    coeff0 = zeros(Npsi^2, length(coeff(1,:)));
                    coeff1 = zeros(Npsi^2, length(coeff(1,:)));
                    coeff2 = zeros(Npsi^2, length(coeff(1,:)));
                    
                    % Build normalization matrix following MATLAB pattern
                    NN = int2_mat(Npsi, Lx);
                    MM = int2_mat(Npsi, Ly);
                    
                    NN = reshape(NN, [Npsi^2, 1]);
                    MM = reshape(MM, [1, Npsi^2]);
                    
                    NN = repmat(NN, [1, Npsi^2]);
                    MM = repmat(MM, [Npsi^2, 1]);
                    
                    nmatr = NN .* MM;
                    nmatr = full(nmatr);
                    nmatr = reshape(nmatr, [Npsi, Npsi, Npsi, Npsi]);
                    nmatr = permute(nmatr, [4, 1, 3, 2]);
                    nmatr = reshape(nmatr, [1, Npsi^4]);
                    nmatr = sparse(nmatr);
                    
                    % Compute normalized coefficients
                    for d = 1:S
                        temp = coeff(:, d);
                        temp = reshape(temp, [Npsi^2, 1]);
                        temp = repmat(temp, [1, Npsi^2]);
                        temp2 = permute(temp, [2, 1]);
                        temp3 = temp .* temp2;
                        temp3 = reshape(temp3, [Npsi^4, 1]);
                        norms = (nmatr * temp3);
                        
                        coeff0(:, d) = coeff(:, d) / sqrt(norms);
                        coeff1(:, d) = coeff(:, d) / sqrt(norms) / sqrt(auto(d));
                        coeff2(:, d) = coeff(:, d) / sqrt(norms) / auto(d);
                    end
                    
                    % Apply final truncation: S = floor(S/2)
                    S_final = floor(S / 2);
                    
                    coeff0_final = coeff0(1:S_final, 1:S_final);
                    coeff1_final = coeff1(1:S_final, 1:S_final);
                    coeff2_final = coeff2(1:S_final, 1:S_final);
                    
                    % Store results for comparison
                    airy_coeff0_results{case_idx} = coeff0_final(:)';  % Flatten
                    airy_coeff1_results{case_idx} = coeff1_final(:)';  % Flatten
                    airy_coeff2_results{case_idx} = coeff2_final(:)';  % Flatten
                    airy_auto_results{case_idx} = auto;  % Sorted cleaned eigenvalues
                    airy_S_results{case_idx} = S_final;  % Final size after truncation
                else
                    % No valid eigenvalues
                    airy_coeff0_results{case_idx} = NaN;
                    airy_coeff1_results{case_idx} = NaN;
                    airy_coeff2_results{case_idx} = NaN;
                    airy_auto_results{case_idx} = NaN;
                    airy_S_results{case_idx} = 0;
                end
                
                % Debug output for first few cases
                if case_idx <= 3
                    fprintf('Case %d: K size=%dx%d, M size=%dx%d, eigenvalues=[%.6e, %.6e, ...]\n', ...
                        case_idx, size(K,1), size(K,2), size(M,1), size(M,2), ...
                        sorted_vals(1), sorted_vals(2));
                end

            catch ME
                fprintf('Error in case %d: %s\n', case_idx, ME.message);
                K_results{case_idx} = NaN;
                M_results{case_idx} = NaN;
                eig_vals_results{case_idx} = NaN;
                eig_vecs_results{case_idx} = NaN;
                airy_coeff0_results{case_idx} = NaN;
                airy_coeff1_results{case_idx} = NaN;
                airy_coeff2_results{case_idx} = NaN;
                airy_auto_results{case_idx} = NaN;
                airy_S_results{case_idx} = 0;
            end
            
            case_idx = case_idx + 1;
        end
    end
end

% Create JSON structure
json_results = struct();
json_results.Npsi_values = Npsi_values;
json_results.Lx_values = Lx_values;
json_results.Ly_values = Ly_values;
json_results.test_cases = test_cases;
json_results.K_results = K_results;
json_results.M_results = M_results;
json_results.eig_vals_results = eig_vals_results;
json_results.eig_vecs_results = eig_vecs_results;

% Add airy_stress_coefficients results
json_results.airy_coeff0_results = airy_coeff0_results;
json_results.airy_coeff1_results = airy_coeff1_results;
json_results.airy_coeff2_results = airy_coeff2_results;
json_results.airy_auto_results = airy_auto_results;
json_results.airy_S_results = airy_S_results;

% Create reference_data directory if it doesn't exist
if ~exist('reference_data', 'dir')
    mkdir('reference_data');
end

% Save as MAT file
save('reference_data/test_K_M_matlab_reference_results.mat', 'json_results');
fprintf('Results saved to MAT file\n');

% Write JSON file
json_text = jsonencode(json_results);

fid = fopen('reference_data/test_K_M_matlab_reference_results.json', 'w');
if fid == -1
    error('Could not open file for writing: test_K_M_matlab_reference_results.json');
end
fprintf(fid, '%s', json_text);
fclose(fid);
fprintf('Results saved to test_K_M_matlab_reference_results.json\n');

% Verify first few cases
fprintf('\nVerification of first few cases:\n');
for i = 1:min(3, num_cases)
    Npsi = test_cases(i, 1);
    Lx = test_cases(i, 2);
    Ly = test_cases(i, 3);
    fprintf('Case %d: Npsi=%d, Lx=%.1f, Ly=%.1f\n', i, Npsi, Lx, Ly);
end

fprintf('\nMATLAB reference generation complete!\n');
