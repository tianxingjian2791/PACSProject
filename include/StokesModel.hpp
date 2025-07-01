#ifndef STOKESMODEL_HPP
#define STOKESMODEL_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>

namespace StokesAMG
{
  using namespace dealii;

  // 抛物线流入速度剖面
  template <int dim>
  class InflowVelocity : public Function<dim>
  {
  public:
    InflowVelocity(double max_velocity)
      : Function<dim>(dim), U(max_velocity)
    {}

    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      AssertDimension(values.size(), dim);
      values = 0.0;
      
      // 在左边界(x=0)处应用抛物线剖面
      const double y = p[1];
      const double y_min = 0.0;
      const double y_max = 0.41;
      
      // 抛物线流速: u_x = 4U * (y - y_min)(y_max - y)/(y_max - y_min)^2
      values[0] = 4.0 * U * y * (y_max - y) / (y_max * y_max);
    }

  private:
    double U; // 最大流入速度
  };

  // 主求解器类
  template <int dim>
  class StokesSolver
  {
  public:
    StokesSolver(double viscosity, double inflow_velocity, double mesh_size, double theta);
    void run(std::ofstream &file);

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    double solve();
    void write_matrix_to_csv(std::ofstream &file, double rho);
    void save_sample(double rho);

    // 辅助函数：生成线性间隔数组
    static std::vector<double> linspace(double start, double end, size_t num_points);

    // 参数
    double viscosity;      // ν (运动粘度)
    double inflow_velocity; // U (最大流入速度)
    double mesh_size;      // h (目标网格尺寸)
    double theta;          // AMG强阈值参数

    // 实际网格尺寸
    double global_h;

    // 网格和有限元
    Triangulation<dim> triangulation;
    FESystem<dim>     velocity_fe;
    FE_Q<dim>         pressure_fe;
    DoFHandler<dim>   dof_handler;

    // 约束
    AffineConstraints<double> constraints;

    // 系统矩阵和向量
    BlockSparsityPattern      sparsity_pattern;
    PETScWrappers::MPI::BlockSparseMatrix system_matrix;
    PETScWrappers::MPI::BlockVector solution;
    PETScWrappers::MPI::BlockVector system_rhs;

    // 求解器控制
    SolverControl solver_control;
  };

  template <int dim>
  StokesSolver<dim>::StokesSolver(double viscosity, double inflow_velocity, 
                                  double mesh_size, double theta)
    : viscosity(viscosity)
    , inflow_velocity(inflow_velocity)
    , mesh_size(mesh_size)
    , theta(theta)
    , velocity_fe(FE_Q<dim>(2), dim)  // Q2 速度元
    , pressure_fe(1)                  // Q1 压力元
    , dof_handler(triangulation)
    , solver_control(1000, 1e-12)     // 最大迭代次数1000，容差1e-12
  {}

  template <int dim>
  void StokesSolver<dim>::make_grid()
  {
    // 创建矩形域 [0, 2.2] x [0, 0.41]
    Point<dim> corner1(0, 0);
    Point<dim> corner2(2.2, 0.41);
    GridGenerator::hyper_rectangle(triangulation, corner1, corner2);
    
    // 创建圆柱孔洞 (圆心(0.2,0.2), 半径0.05)
    Point<dim> center(0.2, 0.2);
    const double radius = 0.05;
    
    // 定义圆柱的流形描述
    static SphericalManifold<dim> manifold(center);
    
    // 创建圆柱孔洞
    GridGenerator::hyper_ball(triangulation, center, radius);
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, manifold);
    
    // 细化网格以达到目标尺寸
    triangulation.refine_global(static_cast<unsigned int>(std::log2(0.14 / mesh_size)));
    
    // 存储实际网格尺寸
    global_h = GridTools::maximal_cell_diameter(triangulation);
  }

  template <int dim>
  void StokesSolver<dim>::setup_system()
  {
    // 混合有限元空间 (速度 + 压力)
    FESystem<dim> fe(velocity_fe, 1, pressure_fe, 1);
    dof_handler.distribute_dofs(fe);
    
    // 输出自由度信息
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (Velocity: " << dof_handler.n_dofs() - pressure_fe.n_dofs()
              << ", Pressure: " << pressure_fe.n_dofs() << ")" << std::endl;

    // 初始化MPI变量 (串行情况下使用单个进程)
    std::vector<IndexSet> locally_owned_dofs;
    std::vector<IndexSet> locally_relevant_dofs;
    DoFTools::count_dofs_per_block(dof_handler, locally_owned_dofs, locally_relevant_dofs);

    // 设置约束 (边界条件)
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    
    // 应用边界条件:
    // - Γ0: 无滑移边界 (速度=0)
    // - Γin: 流入边界 (抛物线速度剖面)
    // - Γout: 自然边界条件 (自由流出)
    InflowVelocity<dim> inflow_profile(inflow_velocity);
    
    // 标记边界: 0=左边界(流入), 1=右边界(流出), 2=其他边界(无滑移)
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    VectorTools::interpolate_boundary_values(dof_handler, 0, inflow_profile, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(dim+1), constraints);
    
    constraints.close();

    // 创建块稀疏模式
    BlockDynamicSparsityPattern dsp(2, 2);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dsp.compress();
    
    // 初始化块矩阵和向量
    system_matrix.reinit(locally_owned_dofs, dsp, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }

  template <int dim>
  void StokesSolver<dim>::assemble_system()
  {
    QGauss<dim> quadrature_formula(velocity_fe.degree + 1);
    
    FEValues<dim> fe_values(velocity_fe, quadrature_formula,
                            update_values | update_gradients |
                            update_JxW_values | update_quadrature_points);
    
    const unsigned int dofs_per_cell = velocity_fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    // 遍历所有单元
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs = 0;
      
      // 组装单元矩阵和右端项
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // 粘性项: ν ∫∇u:∇v dx
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int component_i = velocity_fe.system_to_component_index(i).first;
          
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            const unsigned int component_j = velocity_fe.system_to_component_index(j).first;
            
            if (component_i == component_j)
            {
              cell_matrix(i, j) += viscosity *
                                   fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
            }
          }
        }
      }
      
      // 将单元贡献添加到全局系统
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                            local_dof_indices,
                                            system_matrix, system_rhs);
    }
    
    // 应用约束并压缩矩阵
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  double StokesSolver<dim>::solve()
  {
    // 设置求解器 (MINRES 用于对称不定系统)
    PETScWrappers::SolverMinRes solver(solver_control, MPI_COMM_WORLD);
    
    // 设置块对角预处理器:
    //   [A 0]
    //   [0 M]
    PETScWrappers::PreconditionBoomerAMG A_preconditioner;
    PETScWrappers::PreconditionBoomerAMG M_preconditioner;
    
    // 配置速度块AMG参数
    PETScWrappers::PreconditionBoomerAMG::AdditionalData A_data;
    A_data.strong_threshold = theta;
    A_data.symmetric_operator = true;
    A_preconditioner.initialize(system_matrix.block(0,0), A_data);
    
    // 配置压力质量矩阵AMG参数
    PETScWrappers::PreconditionBoomerAMG::AdditionalData M_data;
    M_data.strong_threshold = 0.25; // 压力块使用固定阈值
    M_preconditioner.initialize(system_matrix.block(1,1), M_data);
    
    // 设置块对角预处理器
    BlockDiagonalPreconditioner<PETScWrappers::PreconditionBase> preconditioner;
    preconditioner.subscribe(A_preconditioner);
    preconditioner.subscribe(M_preconditioner);
    
    // 求解系统
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    
    // 计算收敛因子 ρ = (||r_k|| / ||r_0||)^{1/k}
    const unsigned int k = solver_control.last_step();
    if (k < 1) {
      std::cerr << "Warning: Insufficient residuals recorded. Returning rho=0." << std::endl;
      return 0.0;
    }
    
    // 计算最终残差范数
    PETScWrappers::MPI::BlockVector residual(system_rhs);
    system_matrix.residual(residual, solution, system_rhs);
    const double final_residual = residual.l2_norm();
    
    // 计算初始残差范数
    const double initial_residual = system_rhs.l2_norm();
    
    return std::pow(final_residual / initial_residual, 1.0 / k);
  }

  template <int dim>
  void StokesSolver<dim>::write_matrix_to_csv(std::ofstream &file, double rho)
  {
    // 只保存速度块矩阵 (A_block)
    auto &velocity_matrix = system_matrix.block(0,0);
    
    const unsigned int m = velocity_matrix.m();
    const unsigned int n = velocity_matrix.n();
    
    // 获取矩阵本地部分
    PetscInt start, end;
    MatGetOwnershipRange(velocity_matrix, &start, &end);
    
    std::vector<PetscInt> row_ptr;
    std::vector<PetscInt> col_ind;
    std::vector<PetscScalar> values;
    
    PetscInt idx = 0;
    for (PetscInt i = start; i < end; i++)
    {
      PetscInt ncols;
      const PetscInt *cols;
      const PetscScalar *vals;
      MatGetRow(velocity_matrix, i, &ncols, &cols, &vals);
      
      row_ptr.push_back(idx);
      for (PetscInt j = 0; j < ncols; j++)
      {
        // 过滤接近零的值
        if (std::abs(vals[j]) > 1e-12)
        {
          col_ind.push_back(cols[j]);
          values.push_back(vals[j]);
          idx++;
        }
      }
      MatRestoreRow(velocity_matrix, i, &ncols, &cols, &vals);
    }
    row_ptr.push_back(idx);
    
    // 写入样本元数据
    file << viscosity << "," << inflow_velocity << "," << global_h << "," << theta << "," << rho;
    
    // 写入矩阵维度信息
    file << "," << m << "," << n << "," << values.size();
    
    // 写入矩阵非零值
    for (const auto &val : values)
      file << "," << val;
    
    // 写入行指针
    for (const auto &ptr : row_ptr)
      file << "," << ptr;
    
    // 写入列索引
    for (const auto &col : col_ind)
      file << "," << col;
    
    file << "\n";
  }

  template <int dim>
  void StokesSolver<dim>::run(std::ofstream &file)
  {
    // 创建网格和系统
    make_grid();
    setup_system();
    assemble_system();
    
    // 求解系统并获取收敛因子
    double rho = solve();
    
    // 保存样本数据
    write_matrix_to_csv(file, rho);
  }

  template <int dim>
  std::vector<double> StokesSolver<dim>::linspace(double start, double end, size_t num_points)
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

  // 数据集生成函数
  void generate_stokes_dataset()
  {
    // 打开输出文件
    std::ofstream file("stokes_dataset.csv");
    
    // 参数范围 (3600个样本 = 3ν × 4U × 6h × 25θ)
    const std::vector<double> viscosities = {0.001, 0.1, 10.0};
    const std::vector<double> inflow_velocities = {0.00001, 0.001, 0.1, 10.0};
    const std::vector<double> mesh_sizes = {0.14, 0.07, 0.035, 0.0175, 0.00875, 0.004375};
    const std::vector<double> theta_values = linspace(0.02, 0.9, 25);
    
    unsigned int sample_count = 0;
    const unsigned int total_samples = viscosities.size() * 
                                      inflow_velocities.size() * 
                                      mesh_sizes.size() * 
                                      theta_values.size();
    
    // CSV文件头
    file << "viscosity,inflow_velocity,mesh_size,theta,rho,matrix_rows,matrix_cols,nnz";
    
    // 遍历所有参数组合
    for (double nu : viscosities) {
      for (double U : inflow_velocities) {
        for (double h : mesh_sizes) {
          for (double theta : theta_values) {
            std::cout << "Processing sample " << (++sample_count) << "/" << total_samples
                      << " (ν=" << nu << ", U=" << U << ", h=" << h << ", θ=" << theta << ")" 
                      << std::endl;
            
            // 创建并运行求解器
            StokesSolver<2> solver(nu, U, h, theta);
            solver.run(file);
          }
        }
      }
    }
    
    std::cout << "Stokes dataset generation complete. Total samples: " << sample_count << std::endl;
  }

} // namespace StokesAMG

#endif // STOKESMODEL_HPP