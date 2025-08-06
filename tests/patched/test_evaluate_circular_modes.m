function test_evaluate_circular_modes()
% TEST_EVALUATE_CIRCULAR_MODES Compare our evaluate_circular_modes function 
% with the original plate_def_circ.m implementation
%
% This test generates mode tables, evaluates modal weights using both methods,
% and compares the results to ensure our implementation is correct.

fprintf('Testing evaluate_circular_modes function...\n');

%% Test Parameters
R = 1.0;           % Plate radius
dr = 0.01;         % Integration step
nu = 0.3;          % Poisson ratio
KR_free = 0;       % Free boundary
KR_clamped = inf;  % Clamped boundary

% Test positions on the plate
test_points = [
    0.0, 0.5;      % theta=0, r=0.5 (center line)
    pi/4, 0.7;     % theta=45deg, r=0.7
    pi/2, 0.9;     % theta=90deg, r=0.9 (near edge)
    pi, 0.3;       % theta=180deg, r=0.3
    3*pi/2, 0.8    % theta=270deg, r=0.8
];

% Boundary conditions to test
BCs = {'free', 'clamped'};
KRs = [KR_free, KR_clamped];

%% Generate Mode Tables
fprintf('Generating mode tables...\n');

% Create simple mode tables for testing
% Format: [Index, xkn, k, n, c, xkn^2]
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

mode_tables = {mode_t_free, mode_t_clamped};

%% Run Tests
total_tests = 0;
passed_tests = 0;
tolerance = 1e-10;  % Very tight tolerance for comparison

for bc_idx = 1:length(BCs)
    BC = BCs{bc_idx};
    KR = KRs(bc_idx);
    mode_t = mode_tables{bc_idx};
    
    fprintf('\n=== Testing %s boundary conditions ===\n', upper(BC));
    
    for pt_idx = 1:size(test_points, 1)
        op = test_points(pt_idx, :);  % [theta, r]
        
        fprintf('Test point %d: theta=%.3f, r=%.3f\n', pt_idx, op(1), op(2));
        
        %% Method 1: Our new function
        weights_new = evaluate_circular_modes(mode_t, nu, KR, op, BC, R, dr);
        
        %% Method 2: Original plate_def_circ.m logic
        weights_original = compute_original_method(mode_t, nu, KR, op, BC, R, dr);
        
        %% Compare results
        Nphi = size(mode_t, 1);
        for ii = 1:Nphi
            total_tests = total_tests + 1;
            
            diff = abs(weights_new(ii) - weights_original(ii));
            max_val = max(abs(weights_new(ii)), abs(weights_original(ii)));
            
            if max_val > 1e-12
                relative_error = diff / max_val;
                test_passed = relative_error < tolerance;
            else
                % Both values are essentially zero
                test_passed = diff < 1e-12;
                relative_error = diff;
            end
            
            if test_passed
                passed_tests = passed_tests + 1;
                status = 'PASS';
            else
                status = 'FAIL';
            end
            
            fprintf('  Mode %d: New=%.8e, Original=%.8e, Error=%.2e [%s]\n', ...
                    ii, weights_new(ii), weights_original(ii), relative_error, status);
        end
    end
end

%% Summary
fprintf('\n=== TEST SUMMARY ===\n');
fprintf('Total tests: %d\n', total_tests);
fprintf('Passed: %d\n', passed_tests);
fprintf('Failed: %d\n', total_tests - passed_tests);
fprintf('Success rate: %.1f%%\n', 100 * passed_tests / total_tests);

if passed_tests == total_tests
    fprintf('✅ ALL TESTS PASSED - evaluate_circular_modes is correct!\n');
else
    fprintf('❌ SOME TESTS FAILED - check implementation\n');
end

end

function weights = compute_original_method(mode_t, nu, KR, op, BC, R, dr)
% COMPUTE_ORIGINAL_METHOD Implement the original plate_def_circ.m logic
% exactly as it appears in the reference code

Nphi = size(mode_t, 1);
Nop = 1;  % Single output point

% Extract mode parameters (following plate_def_circ.m pattern)
c_t = mode_t(1:Nphi, 5);
k_t = mode_t(1:Nphi, 3)';
xkn = mode_t(1:Nphi, 2)';  % Note: using xkn directly, not sqrt(om)

weights = zeros(Nphi, 1);

switch BC
    case {'free', 'elastic'}
        % Following plate_def_circ.m lines 327-375
        rp = zeros(Nop, Nphi);
        J0 = zeros(1, Nphi);
        J1 = zeros(1, Nphi);
        J2 = zeros(1, Nphi);
        I0 = zeros(1, Nphi);
        I1 = zeros(1, Nphi);
        I2 = zeros(1, Nphi);
        JJ0 = zeros(Nop, Nphi);
        II0 = zeros(Nop, Nphi);
        Kkn = zeros(1, Nphi);
        Jtild = zeros(1, Nphi);
        Itild = zeros(1, Nphi);

        for ii = 1:Nphi
            J0(ii) = besselj(k_t(ii), xkn(ii));
            J1(ii) = besselj(k_t(ii)-1, xkn(ii));
            J2(ii) = besselj(k_t(ii)-2, xkn(ii));

            I0(ii) = besseli(k_t(ii), xkn(ii));
            I1(ii) = besseli(k_t(ii)-1, xkn(ii));
            I2(ii) = besseli(k_t(ii)-2, xkn(ii));

            Jtild(ii) = xkn(ii)^2*J2(ii) + ((nu-2*k_t(ii)+1)*xkn(ii) + KR)*J1(ii) + (k_t(ii)*(k_t(ii)+1)*(1-nu)-KR*k_t(ii))*J0(ii);
            Itild(ii) = xkn(ii)^2*I2(ii) + ((nu-2*k_t(ii)+1)*xkn(ii) + KR)*I1(ii) + (k_t(ii)*(k_t(ii)+1)*(1-nu)-KR*k_t(ii))*I0(ii);

            JJ0(:, ii) = besselj(k_t(ii), xkn(ii)*op(:, 2));
            II0(:, ii) = besseli(k_t(ii), xkn(ii)*op(:, 2));

            rp(:, ii) = (JJ0(:, ii) - (Jtild(ii)*II0(:, ii)/(Itild(ii)))) .* cos(k_t(ii)*op(:, 1)-(c_t(ii)-1)/2*pi);

            % Normalization using norm_modes
            Kkn(ii) = norm_modes(k_t(ii), xkn(ii), R, dr, nu, KR, BC);

            rp(:, ii) = rp(:, ii) * Kkn(ii);
        end
        
        weights = rp';  % Convert to column vector

    case 'clamped'
        % Following plate_def_circ.m lines 377+ (clamped case)
        rp = zeros(Nop, Nphi);
        Jkn = zeros(1, Nphi);
        Ikn = zeros(1, Nphi);
        Kkn = zeros(1, Nphi);

        for ii = 1:Nphi
            Jkn(ii) = besselj(k_t(ii), xkn(ii));
            Ikn(ii) = besseli(k_t(ii), xkn(ii));

            JJ0 = besselj(k_t(ii), xkn(ii)*op(:, 2));
            II0 = besseli(k_t(ii), xkn(ii)*op(:, 2));

            rp(:, ii) = (JJ0*Ikn(ii) - Jkn(ii)*II0) .* cos(k_t(ii)*op(:, 1)-(c_t(ii)-1)/2*pi);

            % Normalization using norm_modes
            Kkn(ii) = norm_modes(k_t(ii), xkn(ii), R, dr, nu, KR, BC);

            rp(:, ii) = rp(:, ii) * Kkn(ii);
        end
        
        weights = rp';  % Convert to column vector
end

end