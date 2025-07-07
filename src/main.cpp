#include "DiffusionModel.hpp"
#include "StokesModel.hpp"
#include "ElasticModel.hpp"

int main(int argc, char* argv[])
{
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

            std::ofstream file("./datasets/test/raw/test1.csv");
            if (std::string(argv[2]) == "train")
            {
                std::ofstream file("./datasets/train/raw/train1.csv");
            }
            generate_dataset(file, std::string(argv[2]));
            
        }
        else if (std::strcmp(argv[1], "E") == 0)
        {
            using namespace AMGElastic;
            std::cout << "Generating the dataset of Elastic equations" << std::endl;

            std::ofstream file("./datasets/test/raw/test2.csv");
            if (std::string(argv[2]) == "train")
            {
                std::ofstream file("./datasets/train/raw/train2.csv");
            }
            generate_dataset(file, std::string(argv[2]));
            
            // ElasticProblem<2> elastic_problem;
            // elastic_problem.run();
        }
        else if (std::strcmp(argv[1], "S") == 0)
        {
            using namespace AMGStokes;
            std::cout << "Generating the dataset of Stokes equations" << std::endl;

            std::ofstream file("./datasets/test/raw/test3.csv");
            if (std::string(argv[2]) == "train")
            {
                std::ofstream file("./datasets/train/raw/train3.csv");
            }
            generate_dataset(file, std::string(argv[2]));  

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

    return 0;
}