#ifndef DIFFUSIONMODEL_HPP
#define DIFFUSIONMODEL_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_control.h> // 添加 SolverControl 头文件
#include <deal.II/lac/petsc_sparse_matrix.h> // PETSc 矩阵
#include <deal.II/lac/petsc_vector.h> // PETSc 向量
#include <deal.II/lac/petsc_solver.h> // PETSc 求解器
#include <deal.II/lac/petsc_precondition.h> // PETSc 预处理器
#include <deal.II/lac/dynamic_sparsity_pattern.h>
// #include <deal.II/lac/petsc_sparse_matrix.templates.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <hdf5.h> // 添加HDF5支持
#include <fstream>
#include <iostream>
#include <cmath>


namespace AMGDiffusion
{
  using namespace dealii;

  // 枚举定义四种扩散系数模式
  enum class DiffusionPattern
  {
    vertical_stripes,    // (a) 垂直条纹
    vertical_stripes2,  // (b) 第二类垂直条纹
    checkerboard2,        // (c) 4x4棋盘格
    checkerboard       // (d) 2x2棋盘格
  };

  // 精确解函数（根据模式选择）
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(DiffusionPattern pattern)
      : Function<dim>(1), pattern(pattern)
    {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      const double x = p[0];
      const double y = p[1];

      switch (pattern)
      {
      case DiffusionPattern::vertical_stripes:
      case DiffusionPattern::vertical_stripes2:
        return std::cos(M_PI * x) * std::cos(M_PI * y);
      case DiffusionPattern::checkerboard2:
      case DiffusionPattern::checkerboard:
        return std::cos(2 * M_PI * x) * std::cos(2 * M_PI * y);
      default:
        AssertThrow(false, ExcNotImplemented());
        return 0;
      }
    }

  private:
    DiffusionPattern pattern;
  };

  // 右端项函数（根据模式选择）
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide(DiffusionPattern pattern)
      : Function<dim>(1), pattern(pattern)
    {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      const double x = p[0];
      const double y = p[1];
      double u_value;

      switch (pattern)
      {
      case DiffusionPattern::vertical_stripes:
      case DiffusionPattern::checkerboard:
        u_value = std::cos(M_PI * x) * std::cos(M_PI * y);
        return 2 * M_PI * M_PI * u_value; // -Δu = 2π²u
      case DiffusionPattern::vertical_stripes2:
      case DiffusionPattern::checkerboard2:
        u_value = std::cos(2 * M_PI * x) * std::cos(2 * M_PI * y);
        return 8 * M_PI * M_PI * u_value; // -Δu = 8π²u
      default:
        AssertThrow(false, ExcNotImplemented());
        return 0;
      }
    }

  private:
    DiffusionPattern pattern;
  };

  // 扩散系数函数
  template <int dim>
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    DiffusionCoefficient(DiffusionPattern pattern, double epsilon)
      : Function<dim>(1), pattern(pattern), epsilon(epsilon)
    {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      const double x = p[0];
      const double y = p[1];
      const double tol = 1e-10;

      switch (pattern)
      {
      case DiffusionPattern::vertical_stripes: // 垂直条纹 (a)
      {
        // 将区域分成4个垂直条带 (-1, -0.5), [-0.5, 0), [0, 0.5), [0.5, 1)
        if (x < -0.0 + tol)
          return 1.0; // 灰色区域
        else
          return std::pow(10.0, epsilon); // 白色区域
      }
        
      case DiffusionPattern::vertical_stripes2: // 垂直条纹 (c)
      {
        // 将区域分成4个垂直条带 (-1, -0.5), [-0.5, 0), [0, 0.5), [0.5, 1)
        if (x < -0.5 + tol)
          return 1.0; // 灰色区域
        else if (x < 0.0 + tol)
          return std::pow(10.0, epsilon); // 白色区域
        else if (x < 0.5 + tol)
          return 1.0; // 灰色区域
        else
          return std::pow(10.0, epsilon); // 白色区域
      }

      case DiffusionPattern::checkerboard2: // 棋盘格 (c)
      {
        // 分成4x4=16块，每块0.5x0.5
        int i = static_cast<int>(std::floor((x + 1.0) / 0.5));
        int j = static_cast<int>(std::floor((y + 1.0) / 0.5));
        i = std::min(i, 3);
        j = std::min(j, 3);

        // 棋盘模式：当(i+j)为偶数时灰色
        if ((i + j) % 2 == 0)
          return 1.0;
        else
          return std::pow(10.0, epsilon);
      }

      case DiffusionPattern::checkerboard: // 中心方块 (d)
      {
        // 分成2x2=4块，每块1.0x1.0
        int i = static_cast<int>(std::floor(x + 1.0));
        int j = static_cast<int>(std::floor(y + 1.0));
        i = std::min(i, 1);
        j = std::min(j, 1);

        // 棋盘模式：当(i+j)为偶数时灰色
        if ((i + j) % 2 == 0)
          return 1.0;
        else
          return std::pow(10.0, epsilon);
      }

      default:
        AssertThrow(false, ExcNotImplemented());
        return 0;
      }
    }

  private:
    DiffusionPattern pattern;
    double epsilon;
  };

  // 主求解器类
  template <int dim>
  class Solver
  {
  public:
    Solver(DiffusionPattern pattern, double epsilon, unsigned int refinement);
    void set_pattern(DiffusionPattern pattern);
    void set_theta(double theta);
    void set_epsilon(double epsilon);
    void set_refinement(unsigned int refinement);
    void run(std::ofstream &file);


    // 辅助函数：生成线性间隔数组
    static std::vector<double> linspace(double start, double end, size_t num_points) 
    {
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

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve(std::ofstream &file);
    // void output_results() const;
    /*
    void save_sample_to_hdf5(const std::string& filename, double theta, double rho, unsigned int sample_index);
    void save_scalar(hid_t group_id, const std::string& name, double value);
    void save_sparse_matrix(hid_t group_id, const std::string& name, const PETScWrappers::MPI::SparseMatrix& matrix);
    template <typename T>
    void save_vector(hid_t group_id, const std::string& name, const std::vector<T>& data);
    */
    void write_matrix_to_csv(const PETScWrappers::MPI::SparseMatrix &matrix, std::ofstream &file, double rho, double h);


    // 模式参数
    DiffusionPattern pattern;
    double theta;
    double epsilon;
    unsigned int refinement;

    // 网格和有限元
    dealii::Triangulation<dim> triangulation;
    dealii::FE_Q<dim> fe;
    dealii::DoFHandler<dim> dof_handler;
    dealii::SolverControl solver_control; // 添加求解器控制对象

    // 约束和矩阵
    dealii::AffineConstraints<double> constraints;
    dealii::PETScWrappers::MPI::SparseMatrix system_matrix;
    dealii::PETScWrappers::MPI::Vector solution;
    // dealii::PETScWrappers::MPI::Vector init_solution;
    dealii::PETScWrappers::MPI::Vector system_rhs;

    // 精确解和右端项
    ExactSolution<dim> exact_solution;
    RightHandSide<dim> right_hand_side;
    DiffusionCoefficient<dim> diffusion_coefficient;
  };

  template <int dim>
  Solver<dim>::Solver(DiffusionPattern pattern, double epsilon, unsigned int refinement)
    : pattern(pattern)
    , epsilon(epsilon)
    , refinement(refinement)
    , fe(1) // Q1 有限元
    , dof_handler(triangulation)
    , solver_control(dof_handler.n_dofs(), 1e-10) // 最大迭代次数1000，容差1e-12
    , exact_solution(pattern)
    , right_hand_side(pattern)
    , diffusion_coefficient(pattern, epsilon)
  {}

  template <int dim>
  void Solver<dim>::make_grid()
  {
    // 生成正方形网格 [-1,1]^dim
    GridGenerator::hyper_cube(triangulation, -1.0, 1.0);
    triangulation.refine_global(refinement);
    // std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
  }

  template <int dim>
  void Solver<dim>::set_pattern(DiffusionPattern pattern)
  {
    this->pattern = pattern;
  }

  template <int dim>
  void Solver<dim>::set_theta(double theta)
  {
    this->theta = theta;
    // std::cout<<this->theta<<std::endl;
  }

  template <int dim>
  void Solver<dim>::set_epsilon(double epsilon)
  {
    this->epsilon = epsilon;
  }

  template <int dim>
  void Solver<dim>::set_refinement(unsigned int refinement)
  {
    refinement = refinement;
  }

  template <int dim>
  void Solver<dim>::setup_system()
  {
    // 设置自由度
    dof_handler.distribute_dofs(fe);
    // std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    // 初始化MPI变量（串行情况下使用单个进程）
    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // 创建动态稀疏模式
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dsp.compress();

    // 初始化矩阵和向量
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    // init_solution = solution;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    // 设置约束（边界条件）
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            exact_solution,
                                            constraints);
    constraints.close();
  }

  template <int dim>
  void Solver<dim>::assemble_system()
  {
    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                           quadrature_formula,
                           update_values | update_gradients |
                           update_JxW_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // 遍历所有单元
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit(cell);

      // 获取当前单元上的扩散系数（假设在单元上为常数）
      const double mu = diffusion_coefficient.value(fe_values.quadrature_point(0));

      // 组装单元矩阵和右端项
      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            cell_matrix(i, j) += mu *
                                 fe_values.shape_grad(i, q_index) *
                                 fe_values.shape_grad(j, q_index) *
                                 fe_values.JxW(q_index);
          }

          // 右端项：f * phi_i
          cell_rhs(i) += (right_hand_side.value(fe_values.quadrature_point(q_index)) *
                          fe_values.shape_value(i, q_index) *
                          fe_values.JxW(q_index));
        }
      }

      // 将单元贡献添加到全局系统
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                            cell_rhs,
                                            local_dof_indices,
                                            system_matrix,
                                            system_rhs);
    }

    // 应用约束并压缩矩阵
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void Solver<dim>::solve(std::ofstream &file)
  {
    // solution = init_solution;
    // 设置求解器参数
    dealii::PETScWrappers::SolverCG solver(solver_control, MPI_COMM_WORLD);
    dealii::PETScWrappers::PreconditionBoomerAMG preconditioner;

    // 配置BoomerAMG参数
    dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
    data.strong_threshold = theta; // 设置强阈值参数θ（可根据需要调整）
    // std::cout<<"strong threshold: "<<data.strong_threshold<<std::endl;
    data.symmetric_operator = true; // 对称算子

    // 初始化AMG预条件子
    preconditioner.initialize(system_matrix, data);

    PETScWrappers::MPI::Vector residual(system_rhs);

    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    double init_r_norm = residual.l2_norm();
    // std::cout<<init_r_norm<<" ";

    // 求解系统
    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    double final_r_norm = residual.l2_norm();  
    // std::cout<<final_r_norm<<std::endl;


    // 打印迭代信息
    // std::cout << "   Solver converged in " << solver_control.last_step()
    //           << " iterations." << std::endl;

    const unsigned int k = solver_control.last_step();
    if (k < 1) {
      std::cerr << "Warning: Insufficient residuals recorded (" 
                << k << "). Returning rho=0." << std::endl;
      return;
  }

    // ρ = (||r_k|| / ||r_0||)^{1/k}
    const double rho = (k > 0) ? std::pow(final_r_norm / init_r_norm, 1.0 / k) : 0.0;
    double h = triangulation.begin_active()->diameter(); // 网格尺寸
    write_matrix_to_csv(system_matrix, file, rho, h);
    

    // 应用约束
    constraints.distribute(solution);

    // return rho;
  }

  // template <int dim>
  // void Solver<dim>::output_results() const
  // {
  //   // 输出结果（可选）
  //   DataOut<dim> data_out;
  //   data_out.attach_dof_handler(dof_handler);
  //   data_out.add_data_vector(solution, "solution");
  //   data_out.build_patches();

  //   std::string pattern_name;
  //   switch (pattern)
  //   {
  //   case DiffusionPattern::vertical_stripes:   pattern_name = "vertical_stripes";   break;
  //   case DiffusionPattern::vertical_stripes2: pattern_name = "vertical_stripes2"; break;
  //   case DiffusionPattern::checkerboard2:       pattern_name = "checkerboard2";       break;
  //   case DiffusionPattern::checkerboard:     pattern_name = "checkerboard";     break;
  //   }

  //   std::ofstream output("solution-" + pattern_name + ".vtk");
  //   data_out.write_vtk(output);
  // }

    
  template <int dim>
  void Solver<dim>::write_matrix_to_csv(const PETScWrappers::MPI::SparseMatrix &matrix,
    std::ofstream &file,
    double rho,
    double h)
  {  
    const unsigned int m = matrix.m();
    const unsigned int n = matrix.n();
  
    PetscInt start, end;
    MatGetOwnershipRange(matrix, &start, &end);
    
    std::vector<PetscInt> row_ptr;
    std::vector<PetscInt> col_ind;
    std::vector<PetscScalar> values;
    
    PetscInt idx = 0;
    for (PetscInt i = start; i < end; i++) {
      PetscInt ncols;
      const PetscInt* cols;
      const PetscScalar* vals;
      MatGetRow(matrix, i, &ncols, &cols, &vals);
      
      row_ptr.push_back(idx);
      double zero_tol = 1e-12;
      for (PetscInt j = 0; j < ncols; j++) {
        // Filter the zeros explicitly
        if (std::abs(vals[j]) > zero_tol){
          col_ind.push_back(cols[j]);
          values.push_back(vals[j]);
          idx++;
        }
        
      }
      MatRestoreRow(matrix, i, &ncols, &cols, &vals);
    }
    row_ptr.push_back(idx);
  
    // 写入 m(rows), n(cols), rho, h, nnz
    file << m << "," << n << "," << theta << "," << rho << "," << h << "," << values.size();
  
    // 写入所有非零值
    for (const auto &val : values)
    file << "," << val;
  
    // 写入所有行索引
    for (const auto &r : row_ptr)
    file << "," << r;
  
    // 写入所有列索引
    for (const auto &c : col_ind)
    file << "," << c;
  
    file << "\n";
  }

  template <int dim>
  void Solver<dim>::run(std::ofstream &file)
  {
    make_grid();
    setup_system();
    assemble_system();     // 重新组装系统
    
    solve(file);

  }

  void generate_dataset(std::ofstream &file)
  {
    // 参数范围 (9600个样本 = 4模式 × 12ε × 25θ × 8网格)
    const std::array<DiffusionPattern, 4> patterns = 
    {{
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
               
              for (double theta : theta_values) 
              {
                Solver<2> solver(pattern, epsilon, refinement);
                // std::cout<<"theta to be set: "<<theta<<std::endl;
                solver.set_theta(theta);
                solver.run(file);
                sample_index++;
                
                // if (sample_index % 100 == 0) {
                // if (sample_index) {
                std::cout << "Generated " << sample_index << "/4800 samples" << std::endl;
                // }
              }
          }
        }
    }
    std::cout << "Dataset generation complete. Total samples: " << sample_index << std::endl;    
  }

} // namespace AMGTest

#endif // DIFUSSIONMODEL_HPP 
