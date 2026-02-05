#include "DiffusionModel.hpp"
#include "StokesModel.hpp"
#include "ElasticModel.hpp"
#include "UnifiedDataGenerator.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>

/**
 * Unified Data Generator for AMG Learning
 *
 * This program generates datasets in three formats:
 *   1. theta_cnn:  Pooled 50x50 matrix images for CNN
 *   2. theta_gnn:  Sparse graphs for GNN theta prediction
 *   3. p_value:    Graphs with C/F splitting and prolongation matrix
 *
 * Usage:
 *   mpirun -np <nprocs> ./generate_unified [D|E|S] [train|test] [--theta-cnn|--theta-gnn|--p-value|--all]
 *
 * Examples:
 *   mpirun -np 8 ./generate_unified D train --all
 *   mpirun -np 4 ./generate_unified E test --p-value
 *   mpirun -np 16 ./generate_unified S train --theta-gnn
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
                          << " [D|E|S] [train|test] [--theta-cnn|--theta-gnn|--p-value|--all]\n"
                          << "\n"
                          << "Arguments:\n"
                          << "  D|E|S:      Problem type (Diffusion, Elastic, Stokes)\n"
                          << "  train|test: Dataset split\n"
                          << "  --theta-cnn:  Generate CNN format (pooled 50x50 matrices)\n"
                          << "  --theta-gnn:  Generate GNN format (sparse graphs)\n"
                          << "  --p-value:    Generate P-value format (with C/F splitting)\n"
                          << "  --all:        Generate all formats\n"
                          << "\n"
                          << "Examples:\n"
                          << "  mpirun -np 8 " << argv[0] << " D train --all\n"
                          << "  mpirun -np 4 " << argv[0] << " E test --p-value\n";
            }
            return 1;
        }

        // Parse arguments
        std::string problem_type = argv[1];
        std::string split = argv[2];
        std::string format_str = argv[3];

        // Determine output format
        UnifiedAMG::OutputFormat output_format;
        if (format_str == "--theta-cnn")
            output_format = UnifiedAMG::OutputFormat::THETA_CNN;
        else if (format_str == "--theta-gnn")
            output_format = UnifiedAMG::OutputFormat::THETA_GNN;
        else if (format_str == "--p-value")
            output_format = UnifiedAMG::OutputFormat::P_VALUE;
        else if (format_str == "--all")
            output_format = UnifiedAMG::OutputFormat::ALL;
        else
        {
            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
                std::cerr << "Unknown format: " << format_str << "\n";
            }
            return 1;
        }

        // Get MPI info
        const int n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
        const int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

        // Create output directory structure
        std::string base_dir = "./datasets/unified/" + split + "/raw";

        // Open output files (only on rank 0)
        std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;

        if (rank == 0)
        {
            std::cout << "\n==============================================\n";
            std::cout << "Unified AMG Data Generator\n";
            std::cout << "==============================================\n";
            std::cout << "Problem type: " << problem_type << "\n";
            std::cout << "Split: " << split << "\n";
            std::cout << "Format: " << format_str << "\n";
            std::cout << "MPI processes: " << n_procs << "\n";
            std::cout << "OpenMP threads per process: " << omp_get_max_threads() << "\n";
            std::cout << "==============================================\n\n";

            // Create directories
            system(("mkdir -p " + base_dir + "/theta_cnn").c_str());
            system(("mkdir -p " + base_dir + "/theta_gnn").c_str());
            system(("mkdir -p " + base_dir + "/p_value").c_str());

            // Open files
            if (output_format == UnifiedAMG::OutputFormat::THETA_CNN ||
                output_format == UnifiedAMG::OutputFormat::ALL)
            {
                std::string filename = base_dir + "/theta_cnn/" + split + "_" + problem_type + ".csv";
                theta_cnn_file.open(filename, std::ios::out | std::ios::trunc);
                if (!theta_cnn_file.is_open())
                {
                    std::cerr << "Failed to open file: " << filename << "\n";
                    return -1;
                }
                std::cout << "Output (CNN): " << filename << "\n";
            }

            if (output_format == UnifiedAMG::OutputFormat::THETA_GNN ||
                output_format == UnifiedAMG::OutputFormat::ALL)
            {
                std::string filename = base_dir + "/theta_gnn/" + split + "_" + problem_type + ".csv";
                theta_gnn_file.open(filename, std::ios::out | std::ios::trunc);
                if (!theta_gnn_file.is_open())
                {
                    std::cerr << "Failed to open file: " << filename << "\n";
                    return -1;
                }
                std::cout << "Output (GNN): " << filename << "\n";
            }

            if (output_format == UnifiedAMG::OutputFormat::P_VALUE ||
                output_format == UnifiedAMG::OutputFormat::ALL)
            {
                std::string filename = base_dir + "/p_value/" + split + "_" + problem_type + ".csv";
                p_value_file.open(filename, std::ios::out | std::ios::trunc);
                if (!p_value_file.is_open())
                {
                    std::cerr << "Failed to open file: " << filename << "\n";
                    return -1;
                }
                std::cout << "Output (P-value): " << filename << "\n";
            }

            std::cout << "\n";
        }

        // Delegate to appropriate problem generator
        if (problem_type == "D")
        {
            using namespace AMGDiffusion;
            std::cout << "Rank " << rank << ": Generating Diffusion equation dataset...\n";

            // Note: The actual generation logic needs to be adapted from existing code
            // This is a placeholder showing the structure

            // TODO: Implement parallel generation with OpenMP within each MPI rank
            // Similar to existing generate_dataset but using UnifiedAMG::generate_unified_sample
        }
        else if (problem_type == "E")
        {
            using namespace AMGElastic;
            std::cout << "Rank " << rank << ": Generating Elastic equation dataset...\n";

            // TODO: Implement parallel generation
        }
        else if (problem_type == "S")
        {
            using namespace AMGStokes;
            std::cout << "Rank " << rank << ": Generating Stokes equation dataset...\n";

            // TODO: Implement parallel generation
        }
        else
        {
            if (rank == 0)
            {
                std::cerr << "Unknown problem type: " << problem_type << "\n"
                          << "Usage: " << argv[0] << " [D|E|S] [train|test] [format]\n";
            }
            return 1;
        }

        // Close files
        if (rank == 0)
        {
            if (theta_cnn_file.is_open())
                theta_cnn_file.close();
            if (theta_gnn_file.is_open())
                theta_gnn_file.close();
            if (p_value_file.is_open())
                p_value_file.close();

            std::cout << "\n==============================================\n";
            std::cout << "Data generation complete!\n";
            std::cout << "==============================================\n";
        }
    }
    catch (std::exception& exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }

    return 0;
}
