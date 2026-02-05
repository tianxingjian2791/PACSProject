#include "DiffusionModel.hpp"
#include "UnifiedDataGenerator.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>

/**
 * Extra-large dataset generator (10,000+ samples)
 */

int main(int argc, char* argv[])
{
    try
    {
        using namespace dealii;

        // Initialize MPI
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        // Check command-line arguments
        if (argc < 4)
        {
            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
                std::cerr << "Usage: " << argv[0]
                          << " [D|E|S] [train|test] [--theta-cnn|--theta-gnn|--p-value]\n";
            }
            return 1;
        }

        // Parse arguments
        std::string problem_type = argv[1];
        std::string split = argv[2];
        std::string format_str = argv[3];

        // Get MPI info
        const int n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
        const int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::cout << "\n==============================================\n";
            std::cout << "Unified AMG Data Generator (EXTRA LARGE)\n";
            std::cout << "Target: 10,000+ samples\n";
            std::cout << "==============================================\n";
            std::cout << "Problem type: " << problem_type << "\n";
            std::cout << "Split: " << split << "\n";
            std::cout << "Format: " << format_str << "\n";
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

        if (problem_type == "D")
        {
            using namespace AMGDiffusion;

            if (rank == 0)
            {
                std::cout << "Generating Extra Large Diffusion dataset...\n\n";

                // Open output files
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
                else
                {
                    std::cerr << "Unknown format: " << format_str << "\n";
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

                // Extra Large Configuration: 10,240 samples
                // 4 patterns × 20 epsilon × 32 theta × 4 refinements = 10,240
                std::vector<DiffusionPattern> patterns = {
                    DiffusionPattern::vertical_stripes,
                    DiffusionPattern::vertical_stripes2,
                    DiffusionPattern::checkerboard,
                    DiffusionPattern::checkerboard2
                };
                std::vector<double> epsilon_values = Solver<2>::linspace(0.0, 9.5, 20);  // 20 values
                std::vector<double> theta_values = Solver<2>::linspace(0.02, 0.9, 32);   // 32 values
                std::vector<unsigned int> refinements = {3, 4, 5, 6};  // 4 levels

                int total_samples = patterns.size() * epsilon_values.size()
                                  * theta_values.size() * refinements.size();

                std::cout << "Dataset Configuration (EXTRA LARGE):\n";
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

                                if (sample_count % 500 == 0)
                                {
                                    auto current_time = std::chrono::high_resolution_clock::now();
                                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                        current_time - start_time).count();

                                    double progress = (100.0 * sample_count) / total_samples;
                                    double rate = (elapsed > 0) ? (sample_count / (double)elapsed) : 0;
                                    int eta = (elapsed > 0 && sample_count > 0) ?
                                             ((total_samples - sample_count) * elapsed / sample_count) : 0;

                                    std::cout << "  Progress: " << sample_count << "/" << total_samples
                                             << " (" << (int)progress << "%)"
                                             << " - " << elapsed << "s elapsed"
                                             << " - " << (int)rate << " samples/s"
                                             << " - ETA: " << eta << "s\n";
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
                std::cout << "Total time: " << total_time << "s (" << (total_time / 60) << " min)\n";
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
