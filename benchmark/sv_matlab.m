function sv_matlab(input_file, num_iterations, use_single_precision, use_tm)
    % Default arguments if not provided
    if nargin < 1
        input_file = "benchmark_input_010.mat";
    end
    
    if nargin < 2
        num_iterations = 50;
    end
    
    if nargin < 3
        use_single_precision = false;
    end

    if nargin < 4
        use_tm = false;
    end
    
    % Close figures and clear command window
    close all;
    clc;
    
    % Load input data
    load(input_file);
    
    % transpose to have the same shape as q
    lambda_mu = lambda_mu.';
    
    % Convert to single precision if requested
    if use_single_precision
        H = single(H);
        B = single(B);
        C = single(C);
        A_inv = single(A_inv);
        modal_excitation_normalised = single(modal_excitation_normalised);
        modal_gains_at_readout = single(modal_gains_at_readout);
        lambda_mu = single(lambda_mu);
        tau_with_norms = single(tau_with_norms);

        disp("tau_with_norms size: " + size(tau_with_norms));
    end
    
    n_modes = size(modal_excitation_normalised, 1);
    T = size(modal_excitation_normalised, 2);
    
    % Initialize timing array
    times_matlab = zeros(num_iterations, 1);
    
    % Run the benchmark multiple times
    for iter = 1:num_iterations
        tic;
        out = run_computation(n_modes, T, H, B, C, A_inv, modal_excitation_normalised, use_tm, lambda_mu, tau_with_norms);
        times_matlab(iter) = toc;
        fprintf('Iteration %d: %.4f seconds\n', iter, times_matlab(iter));
    end
    
    % output at a single position
    out_pos_matlab = out.' * modal_gains_at_readout;
    
    % Add precision info to output filename
    if use_single_precision
        precision_str = "single";
    else
        precision_str = "double";
    end
    
    % Add TM info to output filename
    if use_tm
        tm_str = "_tm";
    else
        tm_str = "";
    end
    
    output_file = sprintf("sv_matlab_output_%03d_%s%s.mat", n_modes, precision_str, tm_str);
    save(output_file, "out_pos_matlab", "times_matlab", "precision_str", "use_tm");
    
    % Calculate and display statistics
    mean_time = mean(times_matlab);
    std_time = std(times_matlab);
    min_time = min(times_matlab);
    max_time = max(times_matlab);
    
    fprintf('\nBenchmark Statistics (Precision: %s, Use TM: %d):\n', precision_str, use_tm);
    fprintf('Mean execution time: %.4f seconds\n', mean_time);
    fprintf('Standard deviation: %.4f seconds\n', std_time);
    fprintf('Minimum time: %.4f seconds\n', min_time);
    fprintf('Maximum time: %.4f seconds\n', max_time);

    % % Optional: Create a simple box plot
    % figure;
    % boxplot(execution_times);
    % title('Execution Time Distribution');
    % ylabel('Time (seconds)');
    % grid on;

    % % Optional: Create a histogram
    % figure;
    % histogram(execution_times);
    % title('Execution Time Histogram');
    % xlabel('Time (seconds)');
    % ylabel('Frequency');
    % grid on;
end

% Function to run the main computation
function out = run_computation(n_modes, T, H, B, C, A_inv, modal_excitation_normalised, use_tm, lambda_mu, tau_with_norms)
    % Initialize with the same precision as the input data
    if isa(H, 'single')
        q = zeros(n_modes, 1, 'single');
        q_prev = zeros(n_modes, 1, 'single');
        out = zeros(n_modes, T, 'single');
    else
        q = zeros(n_modes, 1);
        q_prev = zeros(n_modes, 1);
        out = zeros(n_modes, T);
    end
    
    H_reshaped = reshape(H, [n_modes * n_modes, n_modes]);
    
    %main loop
    for i=1:T
        if use_tm
            nl = lambda_mu .* q .* (tau_with_norms * (q.^2));
        else
            t0 = H_reshaped*q;
            t0 = reshape(t0, [n_modes, n_modes]);
            t2 = t0*q;
            nl = t0.'*t2;
        end
    
        q_next = B.*q + C.*q_prev - A_inv.*nl + modal_excitation_normalised(:, i);
        q_prev = q;
        q = q_next;
    
        out(:, i) = q;
    end
end
