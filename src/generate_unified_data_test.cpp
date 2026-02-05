#include "DiffusionModel.hpp"
#include "UnifiedDataGenerator.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>

/**
 * Simplified test version of unified data generator
 * This version generates a small number of samples for testing
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
                          << " [D|E|S] [train|test] [--theta-cnn|--theta-gnn|--p-value|--all]\n";
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
            std::cout << "Unified AMG Data Generator (Test Version)\n";
            std::cout << "==============================================\n";
            std::cout << "Problem type: " << problem_type << "\n";
            std::cout << "Split: " << split << "\n";
            std::cout << "Format: " << format_str << "\n";
            std::cout << "MPI processes: " << n_procs << "\n";
            std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
            std::cout << "\nNOTE: This is a simplified test version.\n";
            std::cout << "      Generating small test samples...\n";
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

        // For testing, just call the original generator with reduced parameters
        if (problem_type == "D")
        {
            using namespace AMGDiffusion;

            if (rank == 0)
            {
                std::cout << "Generating Diffusion test samples...\n\n";

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
                    file.open(base_path + "theta_cnn/" + split + "_D.csv",
                             std::ios::out | std::ios::trunc);
                    std::cout << "Note: For --all, generating theta_cnn format only in test mode\n";
                }

                if (!file.is_open())
                {
                    std::cerr << "Failed to open output file\n";
                    return -1;
                }

                // Generate a small test dataset
                // Use simple parameters for quick testing
                std::vector<DiffusionPattern> patterns = {
                    DiffusionPattern::vertical_stripes
                };
                std::vector<double> epsilon_values = {1.0, 2.0};
                std::vector<double> theta_values = {0.25, 0.5};
                std::vector<unsigned int> refinements = {3};  // Small grid for testing

                // Determine output format
                OutputFormat output_format = OutputFormat::THETA_GNN;  // default
                if (format_str == "--theta-cnn")
                {
                    output_format = OutputFormat::THETA_CNN;
                }
                else if (format_str == "--theta-gnn")
                {
                    output_format = OutputFormat::THETA_GNN;
                }
                else if (format_str == "--p-value")
                {
                    output_format = OutputFormat::P_VALUE;
                }

                int sample_count = 0;

                for (auto pattern : patterns)
                {
                    std::cout << "Pattern: vertical_stripes\n";

                    for (double epsilon : epsilon_values)
                    {
                        for (unsigned int refinement : refinements)
                        {
                            for (double theta : theta_values)
                            {
                                sample_count++;
                                std::cout << "  Sample " << sample_count << ": "
                                         << "ε=" << epsilon << ", "
                                         << "θ=" << theta << ", "
                                         << "ref=" << refinement << std::endl;

                                // Create solver
                                Solver<2> solver(pattern, epsilon, refinement);
                                solver.set_theta(theta);

                                // Run solver with specified format
                                solver.run(file, output_format);
                            }
                        }
                    }
                }

                file.close();
                std::cout << "\nGenerated " << sample_count << " test samples\n";
            }
        }
        else if (problem_type == "E")
        {
            if (rank == 0)
            {
                std::cout << "Elastic equation generation not yet implemented in test version\n";
                std::cout << "Please use Diffusion (D) for testing\n";
            }
        }
        else if (problem_type == "S")
        {
            if (rank == 0)
            {
                std::cout << "Stokes equation generation not yet implemented in test version\n";
                std::cout << "Please use Diffusion (D) for testing\n";
            }
        }
        else
        {
            if (rank == 0)
            {
                std::cerr << "Unknown problem type: " << problem_type << "\n";
            }
            return 1;
        }

        if (rank == 0)
        {
            std::cout << "\n==============================================\n";
            std::cout << "Test generation complete!\n";
            std::cout << "==============================================\n";
            std::cout << "\nOutput files in: ./datasets/unified/" << split << "/raw/\n\n";
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
