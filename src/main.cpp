#include "DiffusionModel.hpp"
#include "StokesModel.hpp"
#include "ElasticModel.hpp"
#include "Pooling.hpp"
#include <fstream>

int main(int argc, char* argv[])
{
    // 示例：5×5矩阵 (COO格式)
    // std::vector<double> val = {1.0, 2.0, 3.0, 4.0, 5.0};
    // std::vector<int> row = {0, 1, 2, 3, 4};
    // std::vector<int> col = {0, 1, 2, 3, 4};
    // int n = 5;  // 原始矩阵尺寸
    int m = 50;  // 池化后尺寸

    std::string input_file_name = "./datasets/train/raw/train1.csv";
    std::string out_file_name = "./datasets/train/raw/train1_cnn.csv";

    std::ifstream input_file(input_file_name);
    if (!input_file.is_open()) 
    {
        std::cerr << "Failed to open file\n";
        return -1;
    }

    std::ofstream out_file;
    out_file.open(out_file_name, std::ios::out | std::ios::trunc);
    if (!out_file.is_open()) 
    {
        std::cerr << "Failed to open file\n";
        return -1;
    }

    // std::vector<std::vector<double>> V;
    // std::vector<std::vector<int>> C;

    try {
        // 执行并行池化（SUM操作）
        // parallel_pooling(val, row, col, n, m, PoolingOp::SUM, V, C);
        
        // 输出结果
        // std::cout << "Pooled Matrix (V):" << std::endl;
        // for (const auto& vec : V) {
        //     for (double d : vec) {
        //         std::cout << d << " ";
        //     }
        //     std::cout << std::endl;
        // }
        
        // std::cout << "\nCount Matrix (C):" << std::endl;
        // for (const auto& vec : C) {
        //     for (int i : vec) {
        //         std::cout << i << " ";
        //     }
        //     std::cout << std::endl;
        // }

        pool_dataset(input_file, out_file, m, PoolingOp::SUM);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    /*
    try
    {
        using namespace dealii;

        // 初始化 MPI（在串行模式下也可以安全调用）
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        
        if (argc < 3){
            std::cerr << "Usage: " << argv[0] << "The first argument must be [D|S|E] and the second argument must be [train|test]\n";
            return 1;
        }


        if (std::strcmp(argv[1], "D") == 0)
        {
            using namespace AMGDiffusion;
            std::cout << "Generating the dataset of Diffusion equations" << std::endl;

            std::ofstream file;
            if (std::string(argv[2]) == "train")
            {
                file.open("./datasets/train/raw/train1.csv", std::ios::out | std::ios::trunc);
                if (!file.is_open()) 
                {
                    std::cerr << "Failed to open file\n";
                    return -1;
                }
            }
            else
            {
                file.open("./datasets/test/raw/test1.csv", std::ios::out | std::ios::trunc);
                if (!file.is_open()) 
                {
                    std::cerr << "Failed to open file\n";
                    return -1;
                }
            }
                
            generate_dataset(file, std::string(argv[2]));
            file.close();
            
        }
        else if (std::strcmp(argv[1], "E") == 0)
        {
            using namespace AMGElastic;
            std::cout << "Generating the dataset of Elastic equations" << std::endl;

            std::ofstream file;
            if (std::string(argv[2]) == "train")
            {
                file.open("./datasets/train/raw/train2.csv", std::ios::out | std::ios::trunc);
                if (!file.is_open()) 
                {
                    std::cerr << "Failed to open file\n";
                    return -1;
                }
            }
            else
            {
                file.open("./datasets/test/raw/test2.csv", std::ios::out | std::ios::trunc);
                if (!file.is_open()) 
                {
                    std::cerr << "Failed to open file\n";
                    return -1;
                }
            }
                
            generate_dataset(file, std::string(argv[2]));
            file.close();
            
            // ElasticProblem<2> elastic_problem;
            // elastic_problem.run();
        }
        else if (std::strcmp(argv[1], "S") == 0)
        {
            using namespace AMGStokes;
            std::cout << "Generating the dataset of Stokes equations" << std::endl;

            // std::ofstream file("./datasets/test/raw/test3.csv");
            std::ofstream file;
            if (std::string(argv[2]) == "train")
            {
                file.open("./datasets/train/raw/train3.csv", std::ios::out | std::ios::trunc);
                if (!file.is_open()) 
                {
                    std::cerr << "Failed to open file\n";
                    return -1;
                }
            }
            else
            {
                file.open("./datasets/test/raw/test3.csv", std::ios::out | std::ios::trunc);
                if (!file.is_open()) 
                {
                    std::cerr << "Failed to open file\n";
                    return -1;
                }
            }
                
            generate_dataset(file, std::string(argv[2]));
            file.close(); 

            // StokesProblem<2> stokes_problem(2, 0.1, 0);
            // stokes_problem.run(file);
        }
        else
        {
            std::cerr << "Unknown option: " << argv[1] << "\n"
                  << "Usage: " << argv[0] << " [D|S|E]\n";
            return 1;
        }
    }
    catch (std::exception &exc)
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
    */

    return 0;
}