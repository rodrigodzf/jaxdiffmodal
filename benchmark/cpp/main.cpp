#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>
#include <matioCpp/matioCpp.h>
#include <Eigen/Dense>
#include "matioCpp/ExogenousConversions.h"
#include <cxxopts.hpp>
#include <filesystem>

// Standard implementation of nonlinear term calculation
template <typename T>
std::vector<T> calculate_nonlinear_term(
    int n_modes,
    const std::vector<std::vector<T>>& H_reshaped,
    const std::vector<T>& q
)
{
    // Calculate t0 = H_reshaped * q
    std::vector<T> t0_flat(n_modes * n_modes, 0.0);
    for (int j = 0; j < n_modes * n_modes; j++)
    {
        for (int k = 0; k < n_modes; k++)
        {
            t0_flat[j] += H_reshaped[j][k] * q[k];
        }
    }

    // Reshape t0 to n_modes x n_modes
    std::vector<std::vector<T>> t0(
        n_modes,
        std::vector<T>(n_modes, 0.0)
    );
    for (int j = 0; j < n_modes; j++)
    {
        for (int k = 0; k < n_modes; k++)
        {
            t0[j][k] = t0_flat[j * n_modes + k];
        }
    }

    // Calculate t2 = t0 * q
    std::vector<T> t2(n_modes, 0.0);
    for (int j = 0; j < n_modes; j++)
    {
        for (int k = 0; k < n_modes; k++) { t2[j] += t0[j][k] * q[k]; }
    }

    // Calculate nl = t0^T * t2
    std::vector<T> nl(n_modes, 0.0);
    for (int j = 0; j < n_modes; j++)
    {
        for (int k = 0; k < n_modes; k++) { nl[j] += t0[k][j] * t2[k]; }
    }

    return nl;
}

// Standard implementation of tension modulation nonlinear term calculation
template <typename T>
std::vector<T> calculate_tm_nonlinear_term(
    int n_modes,
    const std::vector<T>& lambda_mu,
    const std::vector<T>& tau_with_norms,
    const std::vector<T>& q
)
{
    // Calculate q^2
    std::vector<T> q_squared(n_modes, 0.0);
    for (int j = 0; j < n_modes; j++) {
        q_squared[j] = q[j] * q[j];
    }

    // Calculate tau_with_norms DOT q^2 (dot product resulting in a scalar)
    T tmp_scalar = 0.0;
    for (int j = 0; j < n_modes; j++) {
        tmp_scalar += tau_with_norms[j] * q_squared[j];
    }

    // Calculate lambda_mu .* q .* (scalar result of dot product)
    std::vector<T> nl(n_modes, 0.0);
    for (int j = 0; j < n_modes; j++) {
        nl[j] = lambda_mu[j] * q[j] * tmp_scalar;
    }

    return nl;
}

// Eigen implementation of nonlinear term calculation
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> calculate_nonlinear_term_eigen(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& H_reshaped,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& q
)
{
    // Calculate t0 = H_reshaped * q
    Eigen::Matrix<T, Eigen::Dynamic, 1> t0_flat = H_reshaped * q;

    // Reshape t0 to n_modes x n_modes
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> t0 =
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(t0_flat.data(), q.size(), q.size());

    // Calculate t2 = t0 * q
    Eigen::Matrix<T, Eigen::Dynamic, 1> t2 = t0 * q;

    // Calculate nl = t0^T * t2
    Eigen::Matrix<T, Eigen::Dynamic, 1> nl = t0.transpose() * t2;

    return nl;
}

// Eigen implementation of tension modulation nonlinear term calculation
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> calculate_tm_nonlinear_term_eigen(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& lambda_mu,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& tau_with_norms,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& q
)
{
    // Calculate q^2
    Eigen::Matrix<T, Eigen::Dynamic, 1> q_squared = q.array().square();
    
    // Calculate tau_with_norms DOT q^2 (dot product resulting in a scalar)
    T tmp_scalar = tau_with_norms.dot(q_squared);
    
    // Calculate lambda_mu .* q .* (scalar result of dot product)
    Eigen::Matrix<T, Eigen::Dynamic, 1> nl = lambda_mu.cwiseProduct(q) * tmp_scalar;
    
    return nl;
}

// Standard implementation of the main computation
template <typename T>
std::vector<std::vector<T>> run_computation_standard(
    int n_modes,
    int Ts,
    const std::vector<std::vector<T>>& H_reshaped,
    const std::vector<T>& B,
    const std::vector<T>& C,
    const std::vector<T>& A_inv,
    const std::vector<std::vector<T>>& modal_excitation_normalised,
    bool use_tm,
    const std::vector<T>& lambda_mu,
    const std::vector<T>& tau_with_norms
)
{
    std::vector<T> q(n_modes, 0.0);
    std::vector<T> q_prev(n_modes, 0.0);
    std::vector<std::vector<T>> out(
        n_modes,
        std::vector<T>(Ts, 0.0)
    );

    // Main loop
    for (int i = 0; i < Ts; i++)
    {
        // Calculate nonlinear term based on use_tm flag
        std::vector<T> nl;
        if (use_tm) {
            nl = calculate_tm_nonlinear_term(n_modes, lambda_mu, tau_with_norms, q);
        } else {
            nl = calculate_nonlinear_term(n_modes, H_reshaped, q);
        }

        // Calculate q_next
        std::vector<T> q_next(n_modes, 0.0);
        for (int j = 0; j < n_modes; j++)
        {
            q_next[j] = B[j] * q[j] + C[j] * q_prev[j] - A_inv[j] * nl[j] +
                        modal_excitation_normalised[j][i];
        }

        // Update q and q_prev
        q_prev = q;
        q = q_next;

        // Store result
        for (int j = 0; j < n_modes; j++) { out[j][i] = q[j]; }
    }

    return out;
}

// Eigen implementation of the main computation
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> run_computation_eigen(
    int Ts,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& H_reshaped,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& B,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& C,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& A_inv,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modal_excitation_normalised,
    bool use_tm,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& lambda_mu,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& tau_with_norms
)
{
    int n_modes = B.size();
    Eigen::Matrix<T, Eigen::Dynamic, 1> q = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_modes);
    Eigen::Matrix<T, Eigen::Dynamic, 1> q_prev = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_modes);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> out(n_modes, Ts);

    // Main loop
    for (int i = 0; i < Ts; i++)
    {
        // Calculate nonlinear term based on use_tm flag
        Eigen::Matrix<T, Eigen::Dynamic, 1> nl;
        if (use_tm) {
            nl = calculate_tm_nonlinear_term_eigen(lambda_mu, tau_with_norms, q);
        } else {
            nl = calculate_nonlinear_term_eigen(H_reshaped, q);
        }

        // Calculate q_next (element-wise operations)
        Eigen::Matrix<T, Eigen::Dynamic, 1> q_next = B.cwiseProduct(q) + C.cwiseProduct(q_prev) -
                                 A_inv.cwiseProduct(nl) +
                                 modal_excitation_normalised.col(i);

        // Update q and q_prev
        q_prev = q;
        q = q_next;

        // Store result
        out.col(i) = q;
    }

    return out;
}

int main(int argc, char* argv[])
{
    // Parse command line arguments
    cxxopts::Options options("cpp_benchmark", "C++ implementation of the benchmark");
    
    options.add_options()
        ("i,input", "Input file path", cxxopts::value<std::string>()->default_value("benchmark_input_010.mat"))
        ("n,iterations", "Number of benchmark iterations", cxxopts::value<int>()->default_value("10"))
        ("o,output", "Output file path", cxxopts::value<std::string>())
        ("s,single", "Use single precision (float) instead of double precision", cxxopts::value<bool>()->default_value("false"))
        ("t,use_tm", "Use tension modulation", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    
    std::string input_file = result["input"].as<std::string>();
    int num_iterations = result["iterations"].as<int>();
    bool use_single_precision = result["single"].as<bool>();
    bool use_tm = result["use_tm"].as<bool>();
    
    // Determine output file path
    std::string output_file;
    if (result.count("output")) {
        output_file = result["output"].as<std::string>();
    } else {
        // Extract the number of modes from the input filename (assuming format benchmark_input_XXX.mat)
        std::filesystem::path input_path(input_file);
        std::string filename = input_path.filename().string();
        size_t pos = filename.find("_");
        pos = filename.find("_", pos + 1);
        std::string n_modes_str = filename.substr(pos + 1, 3);
        
        // Create output filename with precision indicator and tm indicator
        std::string precision_str = use_single_precision ? "float" : "double";
        std::string tm_str = use_tm ? "_tm" : "";
        output_file = "sv_cpp_output_" + n_modes_str + "_" + precision_str + tm_str + ".mat";
    }
    
    std::cout << "Running benchmark with input file: " << input_file << std::endl;
    std::cout << "Number of iterations: " << num_iterations << std::endl;
    std::cout << "Precision: " << (use_single_precision ? "single (float)" : "double") << std::endl;
    std::cout << "Use tension modulation: " << (use_tm ? "true" : "false") << std::endl;
    std::cout << "Output will be saved to: " << output_file << std::endl;

    // Load data from MATLAB file
    matioCpp::File file(input_file);

    // Extract variables
    matioCpp::MultiDimensionalArray<double> modal_excitation =
        file.read("modal_excitation_normalised")
            .asMultiDimensionalArray<double>();
    matioCpp::MultiDimensionalArray<double> H_mat =
        file.read("H").asMultiDimensionalArray<double>();
    matioCpp::Vector<double> B_mat = file.read("B").asVector<double>();
    matioCpp::Vector<double> C_mat = file.read("C").asVector<double>();
    matioCpp::Vector<double> A_inv_mat =
        file.read("A_inv").asVector<double>();
    matioCpp::Vector<double> modal_gains_at_readout =
        file.read("modal_gains_at_readout").asVector<double>();
    
    // Load tension modulation specific variables
    matioCpp::Vector<double> lambda_mu_mat = file.read("lambda_mu").asVector<double>();
    matioCpp::Vector<double> tau_with_norms_mat = file.read("tau_with_norms").asVector<double>();

    // Get dimensions
    int n_modes = modal_excitation.dimensions()[0];
    int T = modal_excitation.dimensions()[1];

    std::cout << "Benchmark parameters: n_modes = " << n_modes << ", T = " << T << std::endl;

    // Run the appropriate precision version
    if (use_single_precision) {
        // Single precision (float) implementation
        // Convert to Eigen matrices with float precision
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> H_eigen = 
            Eigen::Map<const Eigen::MatrixXd>(H_mat.data(), n_modes * n_modes, n_modes).cast<float>();
        
        Eigen::Matrix<float, Eigen::Dynamic, 1> B_eigen = matioCpp::to_eigen(B_mat).cast<float>();
        Eigen::Matrix<float, Eigen::Dynamic, 1> C_eigen = matioCpp::to_eigen(C_mat).cast<float>();
        Eigen::Matrix<float, Eigen::Dynamic, 1> A_inv_eigen = matioCpp::to_eigen(A_inv_mat).cast<float>();
        Eigen::Matrix<float, Eigen::Dynamic, 1> modal_gains_at_readout_eigen = 
            matioCpp::to_eigen(modal_gains_at_readout).cast<float>();
            
        // Convert tension modulation variables
        Eigen::Matrix<float, Eigen::Dynamic, 1> lambda_mu_eigen = matioCpp::to_eigen(lambda_mu_mat).cast<float>();
        Eigen::Matrix<float, Eigen::Dynamic, 1> tau_with_norms_eigen = matioCpp::to_eigen(tau_with_norms_mat).cast<float>();

        // Convert modal_excitation to Eigen matrix with float precision
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> modal_excitation_eigen = 
            matioCpp::to_eigen(modal_excitation).cast<float>();

        // Benchmark Eigen implementation with float precision
        std::cout << "\nBenchmarking Eigen implementation (single precision):" << std::endl;
        std::vector<double> eigen_times(num_iterations);

        for (int iter = 0; iter < num_iterations; iter++) {
            auto start = std::chrono::high_resolution_clock::now();

            auto out = run_computation_eigen<float>(
                T,
                H_eigen,
                B_eigen,
                C_eigen,
                A_inv_eigen,
                modal_excitation_eigen,
                use_tm,
                lambda_mu_eigen,
                tau_with_norms_eigen
            );

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            eigen_times[iter] = elapsed.count();

            std::cout << "Iteration " << (iter + 1) << ": " << eigen_times[iter]
                    << " seconds" << std::endl;
        }

        // Run a final time to get the result
        auto out = run_computation_eigen<float>(
            T,
            H_eigen,
            B_eigen,
            C_eigen,
            A_inv_eigen,
            modal_excitation_eigen,
            use_tm,
            lambda_mu_eigen,
            tau_with_norms_eigen
        );

        // Calculate statistics for Eigen implementation
        double eigen_mean =
            std::accumulate(eigen_times.begin(), eigen_times.end(), 0.0) /
            num_iterations;
        double eigen_variance = 0.0;
        for (double time : eigen_times) {
            eigen_variance += (time - eigen_mean) * (time - eigen_mean);
        }
        eigen_variance /= num_iterations;
        double eigen_stddev = std::sqrt(eigen_variance);
        double eigen_min =
            *std::min_element(eigen_times.begin(), eigen_times.end());
        double eigen_max =
            *std::max_element(eigen_times.begin(), eigen_times.end());

        // Display statistics for Eigen implementation
        std::cout << "\nEigen Implementation Statistics (single precision, use_tm=" << (use_tm ? "true" : "false") << "):" << std::endl;
        std::cout << "Mean execution time: " << eigen_mean << " seconds" << std::endl;
        std::cout << "Standard deviation: " << eigen_stddev << " seconds" << std::endl;
        std::cout << "Minimum time: " << eigen_min << " seconds" << std::endl;
        std::cout << "Maximum time: " << eigen_max << " seconds" << std::endl;

        // Get the output at a single position
        Eigen::Matrix<float, Eigen::Dynamic, 1> out_pos_cpp = out.transpose() * modal_gains_at_readout_eigen;
        
        // Convert back to double for saving
        Eigen::VectorXd out_pos_cpp_double = out_pos_cpp.cast<double>();
        
        // Save the output to a file
        matioCpp::File file_output = matioCpp::File::Create(output_file);
        auto out_cpp = matioCpp::make_variable("out_pos_cpp", out_pos_cpp_double);
        auto times_cpp = matioCpp::make_variable("times_cpp", eigen_times);
        auto precision_var = matioCpp::make_variable("precision", std::string("single"));
        auto use_tm_var = matioCpp::make_variable("use_tm", use_tm);
        file_output.write(out_cpp);
        file_output.write(times_cpp);
        file_output.write(precision_var);
        file_output.write(use_tm_var);
    } else {
        // Double precision implementation
        // Convert to Eigen matrices for Eigen implementation
        Eigen::Map<const Eigen::MatrixXd> H_eigen(
            H_mat.data(),
            n_modes * n_modes,
            n_modes
        );

        Eigen::VectorXd B_eigen = matioCpp::to_eigen(B_mat);
        Eigen::VectorXd C_eigen = matioCpp::to_eigen(C_mat);
        Eigen::VectorXd A_inv_eigen = matioCpp::to_eigen(A_inv_mat);
        Eigen::VectorXd modal_gains_at_readout_eigen =
            matioCpp::to_eigen(modal_gains_at_readout);
            
        // Convert tension modulation variables
        Eigen::VectorXd lambda_mu_eigen = matioCpp::to_eigen(lambda_mu_mat);
        Eigen::VectorXd tau_with_norms_eigen = matioCpp::to_eigen(tau_with_norms_mat);

        // Convert modal_excitation to Eigen matrix
        Eigen::MatrixXd modal_excitation_eigen = matioCpp::to_eigen(modal_excitation);

        // Benchmark Eigen implementation
        std::cout << "\nBenchmarking Eigen implementation (double precision):" << std::endl;
        std::vector<double> eigen_times(num_iterations);

        for (int iter = 0; iter < num_iterations; iter++) {
            auto start = std::chrono::high_resolution_clock::now();

            auto out = run_computation_eigen<double>(
                T,
                H_eigen,
                B_eigen,
                C_eigen,
                A_inv_eigen,
                modal_excitation_eigen,
                use_tm,
                lambda_mu_eigen,
                tau_with_norms_eigen
            );

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            eigen_times[iter] = elapsed.count();

            std::cout << "Iteration " << (iter + 1) << ": " << eigen_times[iter]
                    << " seconds" << std::endl;
        }

        // Run a final time to get the result
        auto out = run_computation_eigen<double>(
            T,
            H_eigen,
            B_eigen,
            C_eigen,
            A_inv_eigen,
            modal_excitation_eigen,
            use_tm,
            lambda_mu_eigen,
            tau_with_norms_eigen
        );

        // Calculate statistics for Eigen implementation
        double eigen_mean =
            std::accumulate(eigen_times.begin(), eigen_times.end(), 0.0) /
            num_iterations;
        double eigen_variance = 0.0;
        for (double time : eigen_times) {
            eigen_variance += (time - eigen_mean) * (time - eigen_mean);
        }
        eigen_variance /= num_iterations;
        double eigen_stddev = std::sqrt(eigen_variance);
        double eigen_min =
            *std::min_element(eigen_times.begin(), eigen_times.end());
        double eigen_max =
            *std::max_element(eigen_times.begin(), eigen_times.end());

        // Display statistics for Eigen implementation
        std::cout << "\nEigen Implementation Statistics (double precision, use_tm=" << (use_tm ? "true" : "false") << "):" << std::endl;
        std::cout << "Mean execution time: " << eigen_mean << " seconds" << std::endl;
        std::cout << "Standard deviation: " << eigen_stddev << " seconds" << std::endl;
        std::cout << "Minimum time: " << eigen_min << " seconds" << std::endl;
        std::cout << "Maximum time: " << eigen_max << " seconds" << std::endl;

        // Get the output at a single position
        Eigen::VectorXd out_pos_cpp = out.transpose() * modal_gains_at_readout_eigen;
        
        // Save the output to a file
        matioCpp::File file_output = matioCpp::File::Create(output_file);
        auto out_cpp = matioCpp::make_variable("out_pos_cpp", out_pos_cpp);
        auto times_cpp = matioCpp::make_variable("times_cpp", eigen_times);
        auto precision_var = matioCpp::make_variable("precision", std::string("double"));
        auto use_tm_var = matioCpp::make_variable("use_tm", use_tm);
        file_output.write(out_cpp);
        file_output.write(times_cpp);
        file_output.write(precision_var);
        file_output.write(use_tm_var);
    }
    
    std::cout << "Results saved to " << output_file << std::endl;

    return 0;
}