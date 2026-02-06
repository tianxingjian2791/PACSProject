/* ---------------------------------------------------------------------
 * generate_amg_data.cpp
 *
 * Unified AMG dataset generator
 *
 * Supports all problem types:
 * - D: Diffusion equations (FEM)
 * - E: Elastic equations (FEM)
 * - S: Stokes equations (FEM)
 * - GL: Graph Laplacian (random graphs)
 * - SC: Spectral Clustering (k-NN graphs)
 *
 * Output formats:
 * - theta-cnn: Pooled 50×50 matrices for CNN
 * - theta-gnn: Sparse CSR graphs for GNN
 * - p-value: Full AMG operators (C/F, P, S matrices)
 * - all: Generate all three formats
 *
 * Usage:
 *   ./generate_amg_data -p <type> -s <split> -f <format> -c <scale> [options]
 * ---------------------------------------------------------------------
 */

#include "../include/DiffusionModel.hpp"
#include "../include/ElasticModel.hpp"
#include "../include/StokesModel.hpp"
#include "../include/GraphLaplacianModel.hpp"
#include "../include/AMGOperators.hpp"
#include "../include/UnifiedDataGenerator.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <sys/stat.h>
#include <omp.h>

// ============================================================================
// Configuration Enums and Structures
// ============================================================================

enum class ProblemType {
    DIFFUSION,
    ELASTIC,
    STOKES,
    GRAPH_LAPLACIAN,
    SPECTRAL_CLUSTERING
};

enum class DatasetSplit {
    TRAIN,
    TEST
};

enum class OutputFormat {
    THETA_CNN,
    THETA_GNN,
    P_VALUE,
    ALL
};

enum class DatasetScale {
    SMALL,
    MEDIUM,
    LARGE,
    XLARGE
};

struct CommandLineArgs {
    ProblemType problem;
    DatasetSplit split;
    OutputFormat format;
    DatasetScale scale;
    std::string output_dir = "./datasets/unified";
    int num_threads = 0;  // 0 = auto (OpenMP default)
    int seed = 42;
    bool verbose = false;
};

// ============================================================================
// Scale Configuration Structures
// ============================================================================

struct FEMScaleConfig {
    std::vector<double> param1_values;  // epsilon for D, E for Elastic, viscosity for S
    std::vector<double> param2_values;  // (optional) nu for E, velocity_degree for S
    std::vector<double> theta_values;
    std::vector<unsigned int> refinements;

    int total_samples() const {
        int count = param1_values.size() * theta_values.size() * refinements.size();
        if (!param2_values.empty()) {
            count *= param2_values.size();
        }
        return count;
    }
};

struct GraphScaleConfig {
    int num_samples;
    int num_points;  // Nodes per graph
};

// ============================================================================
// Utility Functions
// ============================================================================

std::vector<double> linspace(double start, double end, size_t num_points) {
    std::vector<double> result;
    if (num_points == 0) return result;
    if (num_points == 1) {
        result.push_back(start);
        return result;
    }

    double step = (end - start) / (num_points - 1);
    for (size_t i = 0; i < num_points; i++) {
        result.push_back(start + i * step);
    }
    return result;
}

void create_directories(const std::string& path) {
    std::string command = "mkdir -p " + path;
    system(command.c_str());
}

std::string problem_type_to_string(ProblemType type) {
    switch (type) {
        case ProblemType::DIFFUSION: return "D";
        case ProblemType::ELASTIC: return "E";
        case ProblemType::STOKES: return "S";
        case ProblemType::GRAPH_LAPLACIAN: return "GL";
        case ProblemType::SPECTRAL_CLUSTERING: return "SC";
        default: return "UNKNOWN";
    }
}

std::string scale_to_string(DatasetScale scale) {
    switch (scale) {
        case DatasetScale::SMALL: return "small";
        case DatasetScale::MEDIUM: return "medium";
        case DatasetScale::LARGE: return "large";
        case DatasetScale::XLARGE: return "xlarge";
        default: return "unknown";
    }
}

// ============================================================================
// Configuration Factory
// ============================================================================

class ConfigFactory {
public:
    static FEMScaleConfig get_diffusion_config(DatasetScale scale, DatasetSplit split) {
        FEMScaleConfig config;

        switch (scale) {
            case DatasetScale::SMALL:
                config.param1_values = linspace(0.5, 9.0, 5);      // epsilon: 5 values
                config.theta_values = linspace(0.1, 0.8, 5);       // theta: 5 values
                config.refinements = {3, 4};                        // 2 levels
                // Total: 5 × 5 × 2 = 50 samples
                break;

            case DatasetScale::MEDIUM:
                config.param1_values = linspace(0.0, 9.5, 10);     // epsilon: 10 values
                config.theta_values = linspace(0.02, 0.9, 15);     // theta: 15 values
                config.refinements = {3, 4, 5};                     // 3 levels
                // Total: 10 × 15 × 3 = 450 samples
                break;

            case DatasetScale::LARGE:
                config.param1_values = linspace(0.0, 9.5, 12);     // epsilon: 12 values
                config.theta_values = linspace(0.02, 0.9, 25);     // theta: 25 values
                config.refinements = {3, 4, 5, 6};                  // 4 levels
                // Total: 12 × 25 × 4 = 1,200 samples
                break;

            case DatasetScale::XLARGE:
                config.param1_values = linspace(0.0, 9.5, 20);     // epsilon: 20 values
                config.theta_values = linspace(0.02, 0.9, 32);     // theta: 32 values
                config.refinements = {3, 4, 5, 6};                  // 4 levels
                // Total: 20 × 32 × 4 = 2,560 samples
                break;
        }

        return config;
    }

    static FEMScaleConfig get_elastic_config(DatasetScale scale, DatasetSplit split) {
        FEMScaleConfig config;

        switch (scale) {
            case DatasetScale::SMALL:
                config.param1_values = {2.5e2, 2.5e4, 2.5e6};      // E: 3 values
                config.param2_values = {0.25, 0.35};                // nu: 2 values
                config.theta_values = linspace(0.1, 0.8, 5);        // theta: 5 values
                config.refinements = {3, 4};                         // 2 levels
                // Total: 3 × 2 × 5 × 2 = 60 samples
                break;

            case DatasetScale::MEDIUM:
                config.param1_values = {2.5, 2.5e2, 2.5e4, 2.5e6}; // E: 4 values
                config.param2_values = {0.25, 0.3, 0.35};           // nu: 3 values
                config.theta_values = linspace(0.02, 0.9, 15);      // theta: 15 values
                config.refinements = {3, 4, 5};                      // 3 levels
                // Total: 4 × 3 × 15 × 3 = 540 samples
                break;

            case DatasetScale::LARGE:
                config.param1_values = {2.5, 2.5e2, 2.5e4, 2.5e6, 2.5e8}; // 5 values
                config.param2_values = {0.20, 0.25, 0.30, 0.35, 0.40};    // 5 values
                config.theta_values = linspace(0.02, 0.9, 25);      // 25 values
                config.refinements = {3, 4, 5, 6};                   // 4 levels
                // Total: 5 × 5 × 25 × 4 = 2,500 samples
                break;

            case DatasetScale::XLARGE:
                config.param1_values = {2.5, 2.5e2, 2.5e4, 2.5e6, 2.5e8}; // 5 values
                config.param2_values = linspace(0.15, 0.45, 7);     // nu: 7 values
                config.theta_values = linspace(0.02, 0.9, 32);      // theta: 32 values
                config.refinements = {3, 4, 5, 6};                   // 4 levels
                // Total: 5 × 7 × 32 × 4 = 4,480 samples
                break;
        }

        return config;
    }

    static FEMScaleConfig get_stokes_config(DatasetScale scale, DatasetSplit split) {
        FEMScaleConfig config;

        switch (scale) {
            case DatasetScale::SMALL:
                config.param1_values = linspace(0.1, 6.1, 6);       // viscosity: 6 values
                config.param2_values = {2};                          // velocity_degree: 1 value
                config.theta_values = linspace(0.1, 0.8, 5);         // theta: 5 values
                config.refinements = {3, 4};                          // 2 levels
                // Total: 6 × 1 × 5 × 2 = 60 samples
                break;

            case DatasetScale::MEDIUM:
                config.param1_values = linspace(0.1, 6.1, 12);      // viscosity: 12 values
                config.param2_values = {2, 3};                       // velocity_degree: 2 values
                config.theta_values = linspace(0.02, 0.9, 15);       // theta: 15 values
                config.refinements = {3, 4, 5};                       // 3 levels
                // Total: 12 × 2 × 15 × 3 = 1,080 samples
                break;

            case DatasetScale::LARGE:
                config.param1_values = linspace(0.1, 6.1, 15);      // viscosity: 15 values
                config.param2_values = {2, 3};                       // velocity_degree: 2 values
                config.theta_values = linspace(0.02, 0.9, 25);       // theta: 25 values
                config.refinements = {3, 4, 5, 6};                    // 4 levels
                // Total: 15 × 2 × 25 × 4 = 3,000 samples
                break;

            case DatasetScale::XLARGE:
                config.param1_values = linspace(0.1, 6.1, 20);      // viscosity: 20 values
                config.param2_values = {2, 3, 4};                    // velocity_degree: 3 values
                config.theta_values = linspace(0.02, 0.9, 32);       // theta: 32 values
                config.refinements = {3, 4, 5, 6};                    // 4 levels
                // Total: 20 × 3 × 32 × 4 = 7,680 samples
                break;
        }

        return config;
    }

    static GraphScaleConfig get_graph_laplacian_config(DatasetScale scale, DatasetSplit split) {
        GraphScaleConfig config;

        switch (scale) {
            case DatasetScale::SMALL:
                config.num_samples = 50;
                config.num_points = 64;
                break;

            case DatasetScale::MEDIUM:
                config.num_samples = 500;
                config.num_points = 128;
                break;

            case DatasetScale::LARGE:
                config.num_samples = 2000;
                config.num_points = 256;
                break;

            case DatasetScale::XLARGE:
                config.num_samples = 10000;
                config.num_points = 512;
                break;
        }

        return config;
    }

    static GraphScaleConfig get_spectral_clustering_config(DatasetScale scale, DatasetSplit split) {
        return get_graph_laplacian_config(scale, split);
    }
};

// ============================================================================
// Unified AMG Data Generator
// ============================================================================

class UnifiedAMGDataGenerator {
public:
    explicit UnifiedAMGDataGenerator(const CommandLineArgs& args)
        : args_(args)
    {
        // Compute output base directory
        std::string split_str = (args_.split == DatasetSplit::TRAIN) ? "train" : "test";
        output_base_ = args_.output_dir + "/" + split_str + "/raw/";

        // Create directories
        create_directories(output_base_);
    }

    void generate() {
        switch (args_.problem) {
            case ProblemType::DIFFUSION:
                generate_diffusion();
                break;
            case ProblemType::ELASTIC:
                generate_elastic();
                break;
            case ProblemType::STOKES:
                generate_stokes();
                break;
            case ProblemType::GRAPH_LAPLACIAN:
                generate_graph_laplacian();
                break;
            case ProblemType::SPECTRAL_CLUSTERING:
                generate_spectral_clustering();
                break;
            default:
                throw std::runtime_error("Unknown problem type");
        }
    }

private:
    void generate_diffusion();
    void generate_elastic();
    void generate_stokes();
    void generate_graph_laplacian();
    void generate_spectral_clustering();

    void open_output_files(std::ofstream& theta_cnn_file,
                          std::ofstream& theta_gnn_file,
                          std::ofstream& p_value_file);

    void close_output_files(std::ofstream& theta_cnn_file,
                           std::ofstream& theta_gnn_file,
                           std::ofstream& p_value_file);

    void write_sample_theta_cnn(const AMGOperators::CSRMatrix& A,
                               double h, double theta, double rho,
                               std::ofstream& file);

    void write_sample_theta_gnn(const AMGOperators::CSRMatrix& A,
                               double h, double theta, double rho,
                               std::ofstream& file);

    void write_sample_p_value(const AMGOperators::CSRMatrix& A,
                             double h, double theta, double rho,
                             std::ofstream& file);

    void write_sample(const AMGOperators::CSRMatrix& A,
                     double h, double theta, double rho,
                     std::ofstream& theta_cnn_file,
                     std::ofstream& theta_gnn_file,
                     std::ofstream& p_value_file);

    void print_progress(int current, int total,
                       std::chrono::steady_clock::time_point start);

    void print_final_summary(int total,
                            std::chrono::steady_clock::time_point start);

    CommandLineArgs args_;
    std::string output_base_;
};

void UnifiedAMGDataGenerator::open_output_files(
    std::ofstream& theta_cnn_file,
    std::ofstream& theta_gnn_file,
    std::ofstream& p_value_file)
{
    std::string problem_suffix = problem_type_to_string(args_.problem);
    std::string split_prefix = (args_.split == DatasetSplit::TRAIN) ? "train" : "test";

    if (args_.format == OutputFormat::THETA_CNN || args_.format == OutputFormat::ALL) {
        std::string path = output_base_ + "theta_cnn/";
        create_directories(path);
        path += split_prefix + "_" + problem_suffix + ".csv";
        theta_cnn_file.open(path);
        if (!theta_cnn_file) {
            throw std::runtime_error("Failed to open theta_cnn output file: " + path);
        }
    }

    if (args_.format == OutputFormat::THETA_GNN || args_.format == OutputFormat::ALL) {
        std::string path = output_base_ + "theta_gnn/";
        create_directories(path);
        path += split_prefix + "_" + problem_suffix + ".csv";
        theta_gnn_file.open(path);
        if (!theta_gnn_file) {
            throw std::runtime_error("Failed to open theta_gnn output file: " + path);
        }
    }

    if (args_.format == OutputFormat::P_VALUE || args_.format == OutputFormat::ALL) {
        std::string path = output_base_ + "p_value/";
        create_directories(path);
        path += split_prefix + "_" + problem_suffix + ".csv";
        p_value_file.open(path);
        if (!p_value_file) {
            throw std::runtime_error("Failed to open p_value output file: " + path);
        }
    }
}

void UnifiedAMGDataGenerator::close_output_files(
    std::ofstream& theta_cnn_file,
    std::ofstream& theta_gnn_file,
    std::ofstream& p_value_file)
{
    if (theta_cnn_file.is_open()) theta_cnn_file.close();
    if (theta_gnn_file.is_open()) theta_gnn_file.close();
    if (p_value_file.is_open()) p_value_file.close();
}

void UnifiedAMGDataGenerator::write_sample_theta_cnn(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    std::ofstream& file)
{
    // Pool matrix to 50x50 using pooling function from Pooling.hpp
    std::vector<std::vector<double>> V;
    std::vector<std::vector<int>> C;
    int pool_size = 50;

    parallel_pooling_csr(A.values, A.col_indices, A.row_ptr,
                        A.n_rows, pool_size, PoolingOp::SUM, V, C);

    // Standardize
    std_normalize(V);

    // Write: n, rho, h, theta, pooled_values (2500 values)
    file << A.n_rows << "," << rho << "," << h << "," << theta;

    for (int i = 0; i < pool_size; ++i) {
        for (int j = 0; j < pool_size; ++j) {
            file << "," << V[i][j];
        }
    }

    file << "\n";
}

void UnifiedAMGDataGenerator::write_sample_theta_gnn(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    std::ofstream& file)
{
    int n = A.n_rows;
    int nnz = A.values.size();

    // Write: n, rho, h, theta, nnz, values, row_ptr, col_indices
    file << n << "," << rho << "," << h << "," << theta << "," << nnz;

    // Write values
    for (const auto& val : A.values) {
        file << "," << val;
    }

    // Write row pointers
    for (const auto& r : A.row_ptr) {
        file << "," << r;
    }

    // Write column indices
    for (const auto& c : A.col_indices) {
        file << "," << c;
    }

    file << "\n";
}

void UnifiedAMGDataGenerator::write_sample_p_value(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    std::ofstream& file)
{
    // Compute AMG operators
    std::vector<int> cf_splitting = AMGOperators::classical_cf_splitting(A, theta);
    AMGOperators::CSRMatrix S = AMGOperators::compute_strength_matrix(A, theta);
    AMGOperators::CSRMatrix P = AMGOperators::compute_baseline_prolongation(A, cf_splitting);

    int n = A.n_rows;
    int nnz_A = A.values.size();
    int nnz_S = S.values.size();
    int nnz_P = P.values.size();

    // Write header: n, rho, h, theta, nnz_A, nnz_S, nnz_P
    file << n << "," << rho << "," << h << "," << theta << ","
         << nnz_A << "," << nnz_S << "," << nnz_P;

    // Write A matrix (values, row_ptr, col_indices)
    for (const auto& val : A.values) file << "," << val;
    for (const auto& r : A.row_ptr) file << "," << r;
    for (const auto& c : A.col_indices) file << "," << c;

    // Write S matrix
    for (const auto& val : S.values) file << "," << val;
    for (const auto& r : S.row_ptr) file << "," << r;
    for (const auto& c : S.col_indices) file << "," << c;

    // Write P matrix
    for (const auto& val : P.values) file << "," << val;
    for (const auto& r : P.row_ptr) file << "," << r;
    for (const auto& c : P.col_indices) file << "," << c;

    // Write C/F splitting
    for (const auto& cf : cf_splitting) file << "," << cf;

    file << "\n";
}

void UnifiedAMGDataGenerator::write_sample(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    std::ofstream& theta_cnn_file,
    std::ofstream& theta_gnn_file,
    std::ofstream& p_value_file)
{
    if (theta_cnn_file.is_open()) {
        write_sample_theta_cnn(A, h, theta, rho, theta_cnn_file);
    }

    if (theta_gnn_file.is_open()) {
        write_sample_theta_gnn(A, h, theta, rho, theta_gnn_file);
    }

    if (p_value_file.is_open()) {
        write_sample_p_value(A, h, theta, rho, p_value_file);
    }
}

void UnifiedAMGDataGenerator::print_progress(
    int current, int total,
    std::chrono::steady_clock::time_point start)
{
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start).count();
    double rate = current / elapsed;
    double eta = (total - current) / rate;

    std::cout << "[" << current << "/" << total << "] "
              << std::fixed << std::setprecision(1)
              << (100.0 * current / total) << "% | "
              << "Rate: " << std::setprecision(1) << rate << " samples/s | "
              << "ETA: " << std::setprecision(0) << eta << "s"
              << std::endl;
}

void UnifiedAMGDataGenerator::print_final_summary(
    int total,
    std::chrono::steady_clock::time_point start)
{
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ Generation complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total samples: " << total << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2)
              << elapsed << "s" << std::endl;
    std::cout << "Average rate: " << std::setprecision(2)
              << (total / elapsed) << " samples/s" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void UnifiedAMGDataGenerator::generate_diffusion() {
    FEMScaleConfig config = ConfigFactory::get_diffusion_config(args_.scale, args_.split);

    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    open_output_files(theta_cnn_file, theta_gnn_file, p_value_file);

    int total = config.total_samples();
    int current = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Diffusion dataset" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << total << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Nested loops over all parameters
    for (double epsilon : config.param1_values) {
        for (unsigned int refinement : config.refinements) {
            for (double theta : config.theta_values) {
                current++;

                // Create solver (new simplified constructor without pattern)
                AMGDiffusion::Solver<2> solver(epsilon, refinement);
                solver.set_theta(theta);

                // Solve (this stores convergence metrics internally)
                std::ofstream dummy_file;  // Solver needs this but we don't use it
                solver.run(dummy_file);

                // Extract results using getter methods
                AMGOperators::CSRMatrix A = solver.get_system_matrix_csr();
                double h = solver.get_mesh_size();
                double rho = solver.get_convergence_factor();

                // Write to appropriate formats
                write_sample(A, h, theta, rho,
                            theta_cnn_file, theta_gnn_file, p_value_file);

                if (args_.verbose && current % 100 == 0) {
                    print_progress(current, total, start_time);
                }
            }
        }
    }

    close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    print_final_summary(current, start_time);
}

void UnifiedAMGDataGenerator::generate_elastic() {
    FEMScaleConfig config = ConfigFactory::get_elastic_config(args_.scale, args_.split);

    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    open_output_files(theta_cnn_file, theta_gnn_file, p_value_file);

    int total = config.total_samples();
    int current = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Elastic dataset" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << total << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Nested loops over parameters
    for (double E : config.param1_values) {
        for (double nu : config.param2_values) {
            // Compute Lamé parameters
            AMGElastic::MaterialProperties material(E, nu);

            for (unsigned int refinement : config.refinements) {
                for (double theta : config.theta_values) {
                    current++;

                    // Create solver
                    AMGElastic::ElasticProblem<2> solver(material.get_lambda(), material.get_mu());
                    solver.set_theta(theta);

                    // Solve
                    std::ofstream dummy_file;
                    solver.run(dummy_file);

                    // Extract results
                    AMGOperators::CSRMatrix A = solver.get_system_matrix_csr();
                    double h = solver.get_mesh_size();
                    double rho = solver.get_convergence_factor();

                    // Write sample
                    write_sample(A, h, theta, rho,
                                theta_cnn_file, theta_gnn_file, p_value_file);

                    if (args_.verbose && current % 100 == 0) {
                        print_progress(current, total, start_time);
                    }
                }
            }
        }
    }

    close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    print_final_summary(current, start_time);
}

void UnifiedAMGDataGenerator::generate_stokes() {
    FEMScaleConfig config = ConfigFactory::get_stokes_config(args_.scale, args_.split);

    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    open_output_files(theta_cnn_file, theta_gnn_file, p_value_file);

    int total = config.total_samples();
    int current = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Stokes dataset" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << total << std::endl;
    std::cout << "========================================\n" << std::endl;

    unsigned int boundary_choice = 1;  // Fixed boundary condition

    for (double viscosity : config.param1_values) {
        for (double velocity_degree_double : config.param2_values) {
            unsigned int velocity_degree = static_cast<unsigned int>(velocity_degree_double);

            for (unsigned int refinement : config.refinements) {
                for (double theta : config.theta_values) {
                    current++;

                    // Create solver
                    AMGStokes::StokesProblem<2> solver(velocity_degree, viscosity, boundary_choice);
                    solver.set_theta(theta);
                    solver.set_init_refinement(refinement);
                    solver.set_n_cycle(1);  // Single cycle for dataset generation

                    // Solve
                    std::ofstream dummy_file;
                    solver.run(dummy_file);

                    // Extract results
                    AMGOperators::CSRMatrix A = solver.get_system_matrix_csr();
                    double h = solver.get_mesh_size();
                    double rho = solver.get_convergence_factor();

                    // Write sample
                    write_sample(A, h, theta, rho,
                                theta_cnn_file, theta_gnn_file, p_value_file);

                    if (args_.verbose && current % 100 == 0) {
                        print_progress(current, total, start_time);
                    }
                }
            }
        }
    }

    close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    print_final_summary(current, start_time);
}

void UnifiedAMGDataGenerator::generate_graph_laplacian() {
    GraphScaleConfig config = ConfigFactory::get_graph_laplacian_config(args_.scale, args_.split);

    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    open_output_files(theta_cnn_file, theta_gnn_file, p_value_file);

    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Graph Laplacian dataset" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << config.num_samples << std::endl;
    std::cout << "Nodes per graph: " << config.num_points << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configure graph generator
    GraphLaplacian::GraphConfig graph_config;
    graph_config.type = GraphLaplacian::GraphType::LOGNORMAL_LAPLACIAN;
    graph_config.num_points = config.num_points;
    graph_config.log_std = 1.0;
    graph_config.seed = args_.seed;

    // Generate samples
    for (int i = 0; i < config.num_samples; ++i) {
        int thread_seed = args_.seed + i;

        GraphLaplacian::GraphLaplacianGenerator generator(graph_config);
        generator.set_seed(thread_seed);

        // Generate matrix
        AMGOperators::CSRMatrix A = generator.generate();

        // Compute mesh size
        double h = GraphLaplacian::compute_mesh_size(A, config.num_points);

        // Find optimal theta
        double rho;
        double theta = GraphLaplacian::find_optimal_theta_for_graph(A, rho);

        // Write sample
        write_sample(A, h, theta, rho,
                    theta_cnn_file, theta_gnn_file, p_value_file);

        if (args_.verbose && (i + 1) % 100 == 0) {
            print_progress(i + 1, config.num_samples, start_time);
        }
    }

    close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    print_final_summary(config.num_samples, start_time);
}

void UnifiedAMGDataGenerator::generate_spectral_clustering() {
    GraphScaleConfig config = ConfigFactory::get_spectral_clustering_config(args_.scale, args_.split);

    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    open_output_files(theta_cnn_file, theta_gnn_file, p_value_file);

    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Spectral Clustering dataset" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << config.num_samples << std::endl;
    std::cout << "Nodes per graph: " << config.num_points << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configure graph generator for spectral clustering
    GraphLaplacian::GraphConfig graph_config;
    graph_config.type = GraphLaplacian::GraphType::SPECTRAL_CLUSTERING;
    graph_config.num_points = config.num_points;
    graph_config.k_neighbors = 10;
    graph_config.sigma = 0.1;
    graph_config.seed = args_.seed;

    // Generate samples
    for (int i = 0; i < config.num_samples; ++i) {
        int thread_seed = args_.seed + i;

        GraphLaplacian::GraphLaplacianGenerator generator(graph_config);
        generator.set_seed(thread_seed);

        // Generate matrix
        AMGOperators::CSRMatrix A = generator.generate();

        // Compute mesh size
        double h = GraphLaplacian::compute_mesh_size(A, config.num_points);

        // Find optimal theta
        double rho;
        double theta = GraphLaplacian::find_optimal_theta_for_graph(A, rho);

        // Write sample
        write_sample(A, h, theta, rho,
                    theta_cnn_file, theta_gnn_file, p_value_file);

        if (args_.verbose && (i + 1) % 100 == 0) {
            print_progress(i + 1, config.num_samples, start_time);
        }
    }

    close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    print_final_summary(config.num_samples, start_time);
}

// ============================================================================
// Command-Line Argument Parsing
// ============================================================================

void print_help() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Unified AMG Data Generator\n";
    std::cout << "========================================\n\n";

    std::cout << "Usage:\n";
    std::cout << "  ./generate_amg_data [OPTIONS]\n\n";

    std::cout << "Required Arguments:\n";
    std::cout << "  -p, --problem TYPE        Problem type: D|E|S|GL|SC\n";
    std::cout << "  -s, --split SPLIT         Dataset split: train|test\n";
    std::cout << "  -f, --format FORMAT       Output format: theta-cnn|theta-gnn|p-value|all\n";
    std::cout << "  -c, --scale SCALE         Dataset scale: small|medium|large|xlarge\n\n";

    std::cout << "Optional Arguments:\n";
    std::cout << "  -o, --output-dir DIR      Output directory (default: ./datasets/unified)\n";
    std::cout << "  -t, --threads NUM         OpenMP threads (default: auto)\n";
    std::cout << "  --seed SEED               Random seed (default: 42)\n";
    std::cout << "  -v, --verbose             Verbose progress output\n";
    std::cout << "  -h, --help                Show this help message\n\n";

    std::cout << "Problem Types:\n";
    std::cout << "  D  - Diffusion equations (uniform coefficient)\n";
    std::cout << "  E  - Elastic equations (linear elasticity)\n";
    std::cout << "  S  - Stokes equations (incompressible flow)\n";
    std::cout << "  GL - Graph Laplacian (Delaunay with lognormal weights)\n";
    std::cout << "  SC - Spectral Clustering (k-NN graphs)\n\n";

    std::cout << "Examples:\n";
    std::cout << "  # Generate small diffusion training set in theta-cnn format\n";
    std::cout << "  ./generate_amg_data -p D -s train -f theta-cnn -c small\n\n";

    std::cout << "  # Generate xlarge graph Laplacian test set in all formats\n";
    std::cout << "  ./generate_amg_data -p GL -s test -f all -c xlarge --threads 16\n\n";

    std::cout << "  # Generate medium elastic training with custom output\n";
    std::cout << "  ./generate_amg_data -p E -s train -f p-value -c medium -o custom_data\n\n";

    std::cout << "========================================\n\n";
}

bool parse_arguments(int argc, char* argv[], CommandLineArgs& args) {
    bool has_problem = false, has_split = false, has_format = false, has_scale = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_help();
            return false;
        }
        else if (arg == "-p" || arg == "--problem") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            std::string type = argv[i];
            if (type == "D") args.problem = ProblemType::DIFFUSION;
            else if (type == "E") args.problem = ProblemType::ELASTIC;
            else if (type == "S") args.problem = ProblemType::STOKES;
            else if (type == "GL") args.problem = ProblemType::GRAPH_LAPLACIAN;
            else if (type == "SC") args.problem = ProblemType::SPECTRAL_CLUSTERING;
            else {
                std::cerr << "Error: Invalid problem type: " << type << std::endl;
                return false;
            }
            has_problem = true;
        }
        else if (arg == "-s" || arg == "--split") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            std::string split = argv[i];
            if (split == "train") args.split = DatasetSplit::TRAIN;
            else if (split == "test") args.split = DatasetSplit::TEST;
            else {
                std::cerr << "Error: Invalid split: " << split << std::endl;
                return false;
            }
            has_split = true;
        }
        else if (arg == "-f" || arg == "--format") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            std::string format = argv[i];
            if (format == "theta-cnn") args.format = OutputFormat::THETA_CNN;
            else if (format == "theta-gnn") args.format = OutputFormat::THETA_GNN;
            else if (format == "p-value") args.format = OutputFormat::P_VALUE;
            else if (format == "all") args.format = OutputFormat::ALL;
            else {
                std::cerr << "Error: Invalid format: " << format << std::endl;
                return false;
            }
            has_format = true;
        }
        else if (arg == "-c" || arg == "--scale") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            std::string scale = argv[i];
            if (scale == "small") args.scale = DatasetScale::SMALL;
            else if (scale == "medium") args.scale = DatasetScale::MEDIUM;
            else if (scale == "large") args.scale = DatasetScale::LARGE;
            else if (scale == "xlarge") args.scale = DatasetScale::XLARGE;
            else {
                std::cerr << "Error: Invalid scale: " << scale << std::endl;
                return false;
            }
            has_scale = true;
        }
        else if (arg == "-o" || arg == "--output-dir") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            args.output_dir = argv[i];
        }
        else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            args.num_threads = std::stoi(argv[i]);
        }
        else if (arg == "--seed") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            args.seed = std::stoi(argv[i]);
        }
        else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        }
        else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if (!has_problem || !has_split || !has_format || !has_scale) {
        std::cerr << "Error: Missing required arguments" << std::endl;
        print_help();
        return false;
    }

    return true;
}

void print_configuration(const CommandLineArgs& args) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Configuration\n";
    std::cout << "========================================\n";
    std::cout << "Problem type: " << problem_type_to_string(args.problem) << "\n";
    std::cout << "Dataset split: " << (args.split == DatasetSplit::TRAIN ? "train" : "test") << "\n";
    std::cout << "Output format: ";
    switch (args.format) {
        case OutputFormat::THETA_CNN: std::cout << "theta-cnn"; break;
        case OutputFormat::THETA_GNN: std::cout << "theta-gnn"; break;
        case OutputFormat::P_VALUE: std::cout << "p-value"; break;
        case OutputFormat::ALL: std::cout << "all"; break;
    }
    std::cout << "\n";
    std::cout << "Scale: " << scale_to_string(args.scale) << "\n";
    std::cout << "Output directory: " << args.output_dir << "\n";
    std::cout << "Random seed: " << args.seed << "\n";
    if (args.num_threads > 0) {
        std::cout << "OpenMP threads: " << args.num_threads << "\n";
    } else {
        std::cout << "OpenMP threads: auto\n";
    }
    std::cout << "Verbose: " << (args.verbose ? "yes" : "no") << "\n";
    std::cout << "========================================\n";
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    // Initialize MPI (needed for FEM solvers)
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // Parse command-line arguments
    CommandLineArgs args;
    if (!parse_arguments(argc, argv, args)) {
        return 1;
    }

    // Set OpenMP threads
    if (args.num_threads > 0) {
        omp_set_num_threads(args.num_threads);
    }

    // Print configuration
    print_configuration(args);

    // Create generator and run
    try {
        UnifiedAMGDataGenerator generator(args);
        generator.generate();
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
