#include "DiffusionModel.hpp"

int main(int argc, char* argv[])
{
    try
    {
    using namespace dealii;
    using namespace AMGTest;

    // 初始化 MPI（在串行模式下也可以安全调用）
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // 参数设置
    /*
    const DiffusionPattern pattern = DiffusionPattern::vertical_stripes; // 选择模式
    const double epsilon = 2.0; // ε参数 (10^ε)
    const unsigned int refinement = 5; // 全局细化次数（网格大小=2^refinement）
    */

    // Generate Dataset
    // Solver<2> solver(pattern, epsilon, refinement);
    // solver.set_theta(0.25);
    // solver.generate_dataset();
    // solver.run();

    /*
    Solver<2> solver2(DiffusionPattern::checkerboard, epsilon, refinement);
    solver2.run();

    Solver<2> solver3(DiffusionPattern::vertical_stripes2, epsilon, refinement);
    solver3.run();

    Solver<2> solver4(DiffusionPattern::checkerboard2, epsilon, refinement);
    solver4.run();
    */

    std::ofstream file("output.csv");
    
    // 参数范围 (9600个样本 = 4模式 × 12ε × 25θ × 8网格)
    const std::array<DiffusionPattern, 4> patterns = {{
        DiffusionPattern::vertical_stripes,
        DiffusionPattern::vertical_stripes2,
        DiffusionPattern::checkerboard2,
        DiffusionPattern::checkerboard
    }};
    
    const std::vector<double> epsilon_values = 
        Solver<2>::linspace(0.0, 9.5, 40); // dataset2
        // Solver<2>::linspace(0.0, 9.5, 12); // dataset1
    
    const std::vector<double> theta_values = 
        Solver<2>::linspace(0.02, 0.9, 45); // dataset2
        // Solver<2>::linspace(0.02, 0.9, 25); // dataset1
    
    const std::vector<unsigned int> refinements = 
        {3, 4, 5, 6}; // dataset2
        // {8};
        // {7};
        // {3, 4, 5, 6}; // dataset1 对应不同网格尺寸
    
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