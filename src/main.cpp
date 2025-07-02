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
        
        if (argc < 2){
            std::cerr << "Usage: " << argv[0] << " [D|S]\n";
            return 1;
        }

        if (std::strcmp(argv[1], "D") == 0)
        {
            using namespace AMGDiffusion;

            std::ofstream file("./datasets/train/raw/train_.csv");
            
            // 参数范围 (9600个样本 = 4模式 × 12ε × 25θ × 8网格)
            const std::array<DiffusionPattern, 4> patterns = {{
                DiffusionPattern::vertical_stripes,
                DiffusionPattern::vertical_stripes2,
                DiffusionPattern::checkerboard2,
                DiffusionPattern::checkerboard
            }};
            
            const std::vector<double> epsilon_values = 
                // Solver<2>::linspace(0.0, 9.5, 40); // dataset2
                Solver<2>::linspace(0.0, 9.5, 12); // dataset1
            
            const std::vector<double> theta_values = 
                // Solver<2>::linspace(0.02, 0.9, 45); // dataset2
                Solver<2>::linspace(0.02, 0.9, 25); // dataset1
            
            const std::vector<unsigned int> refinements = 
                // {3, 4, 5, 6}; // dataset2
                // {8};
                // {7}; // dataset1: test
                {3, 4, 5, 6}; // dataset1: train
            
            unsigned int sample_index = 0;
            
            // 遍历所有参数组合
            for (auto pattern : patterns) {

                // set_pattern(pattern);
                switch (pattern)
                {
                case DiffusionPattern::vertical_stripes: std::cout<<"vertical_stripes"<<std::endl; break;
                case DiffusionPattern::vertical_stripes2: std::cout<<"vertical_stripes2"<<std::endl; break;
                case DiffusionPattern::checkerboard2: std::cout<<"checkerboard2"<<std::endl; break;
                case DiffusionPattern::checkerboard: std::cout<<"checkerboard"<<std::endl; break;
                }
                
                
                for (double epsilon : epsilon_values) {
                // set_epsilon(epsilon);
                
                for (unsigned int refinement : refinements) {
                    // set_refinement(refinement);
                    
                    // 重新生成网格和系统
                    // triangulation.clear();
                    // make_grid();
                    // setup_system();
                    // assemble_system();
                    
                    for (double theta : theta_values) {
                    Solver<2> solver(pattern, epsilon, refinement);
                    // std::cout<<"theta to be set: "<<theta<<std::endl;
                    solver.set_theta(theta);
                    solver.run(file);
                    sample_index++;
                    
                    // if (sample_index % 100 == 0) {
                    // if (sample_index) {
                    std::cout << "Generated " << sample_index << "/28800 samples" << std::endl;
                    // }
                    }
                }
                }
            }
            std::cout << "Dataset generation complete. Total samples: " << sample_index << std::endl;

        }
        else if (std::strcmp(argv[1], "E") == 0)
        {
            using namespace AMGElastic;
            std::cout << "Generating the dataset of Elastic equations" << std::endl;
            // generate_stokes_dataset();
            ElasticProblem<2> elastic_problem;
            elastic_problem.run();
        }
        else if (std::strcmp(argv[1], "S") == 0)
        {
            using namespace AMGStokes;
            std::cout << "Generating the dataset of Stokes equations" << std::endl;
            // generate_stokes_dataset();      
            StokesProblem<2> stokes_problem(2);
            stokes_problem.run();
        }
        else
        {
            std::cerr << "Unknown option: " << argv[1] << "\n"
                  << "Usage: " << argv[0] << " [D|S]\n";
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