#include "DiffusionModel.hpp"
#include "UnifiedDataGenerator.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>

/**
 * Production data generator for unified AMG learning
 * Generates configurable dataset sizes
 */

int main(int argc, char* argv[])
{
    try
    {
        using namespace dealii;

        // Initialize MPI
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        // Check command-line arguments
        if (argc < 5)
        {
            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
                std::cerr << "Usage: " << argv[0]
                          << " [D|E|S] [train|test] [--theta-cnn|--theta-gnn|--p-value|--all] [small|medium|large]\n";
                std::cerr << "\nDataset sizes:\n";
                std::cerr << "  small:  ~100 samples (quick testing)\n";
                std::cerr << "  medium: ~1200 samples (validation)\n";
                std::cerr << "  large:  ~4800 samples (full training)\n";
            }
            return 1;
        }

        // Parse arguments
        std::string problem_type = argv[1];
        std::string split = argv[2];
        std::string format_str = argv[3];
        std::string size_str = argv[4];

        // Get MPI info
        const int n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
        const int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::cout << "\n==============================================\n";
            std::cout << "Unified AMG Data Generator (Production)\n";
            std::cout << "==============================================\n";
            std::cout << "Problem type: " << problem_type << "\n";
            std::cout << "Split: " << split << "\n";
            std::cout << "Format: " << format_str << "\n";
            std::cout << "Size: " << size_str << "\n";
            std::cout << "MPI processes: " << n_procs << "\n";
            std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
            std::cout << "==============================================\n\n";

            // Create output directories
            std::string cmd;
            cmd = "mkdir -p ./datasets/unified/" + split + "/raw/theta_cnn";
            system(cmd.c_str());
            cmd = "mkdir -p ./datasets/unified/" + split + "/raw/theta_gnn";
            system(cmd.c_str());
            cmd = "mkdir -p ./datasets/unified/" + split + "/raw/p_value";
            system(cmd.c_str());
        }

        // Configure dataset parameters based on size
        if (problem_type == "D")
        {
            using namespace AMGDiffusion;

            if (rank == 0)
            {
                std::cout << "Generating Diffusion samples...\n\n";

                // Open output files based on format
                std::ofstream file;
                std::string base_path = "./datasets/unified/" + split + "/raw/";

                if (format_str == "--theta-cnn")
                {
                    file.open(base_path + "theta_cnn/" + split + "_D.csv",
                             std::ios::out | std::ios::trunc);
                }
                else if (format_str == "--theta-gnn")
                {
                    file.open(base_path + "theta_gnn/" + split + "_D.csv",
                             std::ios::out | std::ios::trunc);
                }
                else if (format_str == "--p-value")
                {
                    file.open(base_path + "p_value/" + split + "_D.csv",
                             std::ios::out | std::ios::trunc);
                }
                else  // --all
                {
                    std::cerr << "Note: --all not supported yet. Use individual formats.\n";
                    return 1;
                }

                if (!file.is_open())
                {
                    std::cerr << "Failed to open output file\n";
                    return -1;
                }

                // Determine output format
                OutputFormat output_format = OutputFormat::THETA_GNN;
                if (format_str == "--theta-cnn")
                    output_format = OutputFormat::THETA_CNN;
                else if (format_str == "--theta-gnn")
                    output_format = OutputFormat::THETA_GNN;
                else if (format_str == "--p-value")
                    output_format = OutputFormat::P_VALUE;

                // Configure parameters based on dataset size
                std::vector<DiffusionPattern> patterns;
                std::vector<double> epsilon_values;
                std::vector<double> theta_values;
                std::vector<unsigned int> refinements;

                if (size_str == "small")
                {
                    // Small: ~100 samples
                    patterns = {DiffusionPattern::vertical_stripes};
                    epsilon_values = {0.5, 1.0, 2.0, 5.0, 9.0};  // 5 values
                    theta_values = Solver<2>::linspace(0.1, 0.8, 5);  // 5 values
                    refinements = {3, 4};  // 2 levels
                    // Total: 1 * 5 * 5 * 2 = 50 samples per split
                }
                else if (size_str == "medium")
                {
                    // Medium: ~1200 samples
                    patterns = {
                        DiffusionPattern::vertical_stripes,
                        DiffusionPattern::checkerboard
                    };
                    epsilon_values = Solver<2>::linspace(0.0, 9.5, 10);  // 10 values
                    theta_values = Solver<2>::linspace(0.02, 0.9, 15);   // 15 values
                    refinements = {3, 4, 5};  // 3 levels
                    // Total: 2 * 10 * 15 * 3 = 900 samples per split
                }
                else if (size_str == "large")
                {
                    // Large: ~4800 samples (full dataset)
                    patterns = {
                        DiffusionPattern::vertical_stripes,
                        DiffusionPattern::vertical_stripes2,
                        DiffusionPattern::checkerboard,
                        DiffusionPattern::checkerboard2
                    };
                    epsilon_values = Solver<2>::linspace(0.0, 9.5, 12);  // 12 values
                    theta_values = Solver<2>::linspace(0.02, 0.9, 25);   // 25 values
                    refinements = {3, 4, 5, 6};  // 4 levels
                    // Total: 4 * 12 * 25 * 4 = 4800 samples
                }
                else
                {
                    std::cerr << "Unknown size: " << size_str << "\n";
                    std::cerr << "Use: small, medium, or large\n";
                    return 1;
                }

                // Calculate total samples
                int total_samples = patterns.size() * epsilon_values.size()
                                  * theta_values.size() * refinements.size();

                std::cout << "Dataset Configuration:\n";
                std::cout << "  Patterns: " << patterns.size() << "\n";
                std::cout << "  Epsilon values: " << epsilon_values.size() << "\n";
                std::cout << "  Theta values: " << theta_values.size() << "\n";
                std::cout << "  Refinement levels: " << refinements.size() << "\n";
                std::cout << "  Total samples: " << total_samples << "\n\n";

                int sample_count = 0;
                auto start_time = std::chrono::high_resolution_clock::now();

                // Generate samples
                for (auto pattern : patterns)
                {
                    std::string pattern_name;
                    switch (pattern)
                    {
                        case DiffusionPattern::vertical_stripes:
                            pattern_name = "vertical_stripes"; break;
                        case DiffusionPattern::vertical_stripes2:
                            pattern_name = "vertical_stripes2"; break;
                        case DiffusionPattern::checkerboard:
                            pattern_name = "checkerboard"; break;
                        case DiffusionPattern::checkerboard2:
                            pattern_name = "checkerboard2"; break;
                    }

                    std::cout << "\nPattern: " << pattern_name << "\n";

                    for (double epsilon : epsilon_values)
                    {
                        for (unsigned int refinement : refinements)
                        {
                            for (double theta : theta_values)
                            {
                                sample_count++;

                                if (sample_count % 100 == 0)
                                {
                                    auto current_time = std::chrono::high_resolution_clock::now();
                                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                        current_time - start_time).count();

                                    std::cout << "  Progress: " << sample_count << "/" << total_samples
                                             << " (" << (100 * sample_count / total_samples) << "%)"
                                             << " - " << elapsed << "s elapsed\n";
                                }

                                // Create and run solver
                                Solver<2> solver(pattern, epsilon, refinement);
                                solver.set_theta(theta);
                                solver.run(file, output_format);
                            }
                        }
                    }
                }

                file.close();

                auto end_time = std::chrono::high_resolution_clock::now();
                auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
                    end_time - start_time).count();

                std::cout << "\n==============================================\n";
                std::cout << "Generation complete!\n";
                std::cout << "==============================================\n";
                std::cout << "Total samples: " << sample_count << "\n";
                std::cout << "Total time: " << total_time << "s\n";
                std::cout << "Rate: " << (sample_count / (double)total_time) << " samples/s\n";
                std::cout << "Output: " << base_path << format_str.substr(2) << "/" << split << "_D.csv\n";
                std::cout << "==============================================\n\n";
            }
        }
        else
        {
            if (rank == 0)
            {
                std::cerr << "Only Diffusion (D) problems supported currently\n";
            }
            return 1;
        }
    }
    catch (std::exception& exc)
    {
        std::cerr << "\n----------------------------------------------------\n";
        std::cerr << "Exception: " << exc.what() << "\n";
        std::cerr << "----------------------------------------------------\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "\n----------------------------------------------------\n";
        std::cerr << "Unknown exception!\n";
        std::cerr << "----------------------------------------------------\n";
        return 1;
    }

    return 0;
}
