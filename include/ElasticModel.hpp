/* ---------------------------------------------------------------------
 *
 * This code origins from the step 17 tutorial program of the library deal.II
 * I modify some code of the program in order to generate the dataset
 * ---------------------------------------------------------------------
 
 *
 * Original Author: Wolfgang Bangerth, University of Texas at Austin, 2000, 2004
 *         Wolfgang Bangerth, Texas A&M University, 2016
 */

 
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <fstream>
#include <iostream>

namespace AMGElastic
{
  using namespace dealii;


  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem(double lambda, double mu);
    void run(std::ofstream &file);
    void set_theta(double theta);

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
    void         setup_system();
    void         assemble_system();
    unsigned int solve(std::ofstream &file);
    void         refine_grid();
    void         output_results(const unsigned int cycle) const;
    void write_matrix_to_csv(const PETScWrappers::MPI::SparseMatrix &matrix, std::ofstream &file, double rho, double h);

    MPI_Comm mpi_communicator;

    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    double lambda_;
    double mu_;
    double theta;

    ConditionalOStream pcout;

    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> hanging_node_constraints;

    PETScWrappers::MPI::SparseMatrix system_matrix;

    PETScWrappers::MPI::Vector solution;
    PETScWrappers::MPI::Vector system_rhs;
  };



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override
    {
      AssertDimension(values.size(), dim);
      Assert(dim >= 2, ExcInternalError());

      Point<dim> point_1, point_2;
      point_1(0) = 0.5;
      point_2(0) = -0.5;

      if (((p - point_1).norm_square() < 0.2 * 0.2) ||
          ((p - point_2).norm_square() < 0.2 * 0.2))
        values(0) = 1;
      else
        values(0) = 0;

      if (p.square() < 0.2 * 0.2)
        values(1) = 1;
      else
        values(1) = 0;
    }

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override
    {
      const unsigned int n_points = points.size();

      AssertDimension(value_list.size(), n_points);

      for (unsigned int p = 0; p < n_points; ++p)
        RightHandSide<dim>::vector_value(points[p], value_list[p]);
    }
  };





  template <int dim>
  ElasticProblem<dim>::ElasticProblem(double lambda, double mu)
    : mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , lambda_(lambda)
    , mu_(mu)
    , pcout(std::cout, (this_mpi_process == 0))
    , fe(FE_Q<dim>(1), dim)
    , dof_handler(triangulation)
  {}




  template <int dim>
  void ElasticProblem<dim>::setup_system()
  {
    GridTools::partition_triangulation(n_mpi_processes, triangulation);

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::subdomain_wise(dof_handler);

    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    hanging_node_constraints,
                                    false);

    const std::vector<IndexSet> locally_owned_dofs_per_proc =
      DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
    const IndexSet locally_owned_dofs =
      locally_owned_dofs_per_proc[this_mpi_process];

    system_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        mpi_communicator);

    solution.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  }




  template <int dim>
  void ElasticProblem<dim>::assemble_system()
  {
    QGauss<dim>   quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);

    Functions::ConstantFunction<dim> lambda(lambda_), mu(mu_);

    RightHandSide<dim>          right_hand_side;
    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim));


    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->subdomain_id() == this_mpi_process)
        {
          cell_matrix = 0;
          cell_rhs    = 0;

          fe_values.reinit(cell);

          lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
          mu.value_list(fe_values.get_quadrature_points(), mu_values);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe.system_to_component_index(i).first;

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const unsigned int component_j =
                    fe.system_to_component_index(j).first;

                  for (unsigned int q_point = 0; q_point < n_q_points;
                      ++q_point)
                    {
                      cell_matrix(i, j) +=
                        ((fe_values.shape_grad(i, q_point)[component_i] *
                          fe_values.shape_grad(j, q_point)[component_j] *
                          lambda_values[q_point]) +
                        (fe_values.shape_grad(i, q_point)[component_j] *
                          fe_values.shape_grad(j, q_point)[component_i] *
                          mu_values[q_point]) +
                        ((component_i == component_j) ?
                            (fe_values.shape_grad(i, q_point) *
                            fe_values.shape_grad(j, q_point) *
                            mu_values[q_point]) :
                            0)) *
                        fe_values.JxW(q_point);
                    }
                }
            }

          right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                            rhs_values);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe.system_to_component_index(i).first;

              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                cell_rhs(i) += fe_values.shape_value(i, q_point) *
                              rhs_values[q_point](component_i) *
                              fe_values.JxW(q_point);
            }

          cell->get_dof_indices(local_dof_indices);
          hanging_node_constraints.distribute_local_to_global(cell_matrix,
                                                              cell_rhs,
                                                              local_dof_indices,
                                                              system_matrix,
                                                              system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            Functions::ZeroFunction<dim>(dim),
                                            boundary_values);
    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, false);
  }


  template <int dim>
  void ElasticProblem<dim>::set_theta(double theta)
  {
    this->theta = theta;
  }


  template <int dim>
  unsigned int ElasticProblem<dim>::solve(std::ofstream &file)
  {
    SolverControl solver_control(solution.size(), 1e-8 * system_rhs.l2_norm());
    PETScWrappers::SolverCG cg(solver_control);

    PETScWrappers::PreconditionBoomerAMG preconditioner;
    PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
    data.strong_threshold = theta; // setup θ
    // std::cout<<"strong threshold: "<<data.strong_threshold<<std::endl;
    data.symmetric_operator = true;

    // Initialize AMG preconditioner
    preconditioner.initialize(system_matrix, data);

    PETScWrappers::MPI::Vector residual(system_rhs);

    // system_matrix.vmult(residual, solution);
    // residual -= system_rhs;
    double init_r_norm = residual.l2_norm();
    std::cout<<init_r_norm<<" ";

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    double final_r_norm = residual.l2_norm();  
    std::cout<<final_r_norm<<std::endl;

    Vector<double> localized_solution(solution);

    hanging_node_constraints.distribute(localized_solution);

    solution = localized_solution;

    const unsigned int k = solver_control.last_step();
    if (k < 1) {
      std::cerr << "Warning: Insufficient residuals recorded (" 
                << k << "). Returning rho=0." << std::endl;
      return 0;
    }

    // ρ = (||r_k|| / ||r_0||)^{1/k}
    const double rho = (k > 0) ? std::pow(final_r_norm / init_r_norm, 1.0 / k) : 0.0;
    double h = triangulation.begin_active()->diameter(); // The step of grids
    write_matrix_to_csv(system_matrix, file, rho, h);

    return solver_control.last_step();
  }



  template <int dim>
  void ElasticProblem<dim>::refine_grid()
  {
    const Vector<double> localized_solution(solution);

    Vector<float> local_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                      QGauss<dim - 1>(fe.degree + 1),
                                      {},
                                      localized_solution,
                                      local_error_per_cell,
                                      ComponentMask(),
                                      nullptr,
                                      MultithreadInfo::n_threads(),
                                      this_mpi_process);

    const unsigned int n_local_cells =
      GridTools::count_cells_with_subdomain_association(triangulation,
                                                        this_mpi_process);
    PETScWrappers::MPI::Vector distributed_all_errors(
      mpi_communicator, triangulation.n_active_cells(), n_local_cells);

    for (unsigned int i = 0; i < local_error_per_cell.size(); ++i)
      if (local_error_per_cell(i) != 0)
        distributed_all_errors(i) = local_error_per_cell(i);
    distributed_all_errors.compress(VectorOperation::insert);


    const Vector<float> localized_all_errors(distributed_all_errors);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    localized_all_errors,
                                                    0.3,
                                                    0.03);
    triangulation.execute_coarsening_and_refinement();
  }



  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
  {
    const Vector<double> localized_solution(solution);

    if (this_mpi_process == 0)
      {
        std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        std::vector<std::string> solution_names;
        switch (dim)
          {
            case 1:
              solution_names.emplace_back("displacement");
              break;
            case 2:
              solution_names.emplace_back("x_displacement");
              solution_names.emplace_back("y_displacement");
              break;
            case 3:
              solution_names.emplace_back("x_displacement");
              solution_names.emplace_back("y_displacement");
              solution_names.emplace_back("z_displacement");
              break;
            default:
              Assert(false, ExcInternalError());
          }

        data_out.add_data_vector(localized_solution, solution_names);

        std::vector<unsigned int> partition_int(triangulation.n_active_cells());
        GridTools::get_subdomain_association(triangulation, partition_int);

        const Vector<double> partitioning(partition_int.begin(),
                                          partition_int.end());

        data_out.add_data_vector(partitioning, "partitioning");

        data_out.build_patches();
        data_out.write_vtk(output);
      }
  }


  template <int dim>
  void ElasticProblem<dim>::write_matrix_to_csv(const PETScWrappers::MPI::SparseMatrix &matrix,
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
  
    // write m(rows), n(cols), rho, h, nnz
    file << m << "," << n << "," << theta << "," << rho << "," << h << "," << values.size();
  
    // write non-zero values
    for (const auto &val : values)
    file << "," << val;
  
    // write row ptrs
    for (const auto &r : row_ptr)
    file << "," << r;
  
    // write col indices
    for (const auto &c : col_ind)
    file << "," << c;
  
    file << "\n";
  }


  template <int dim>
  void ElasticProblem<dim>::run(std::ofstream &file)
  {
    for (unsigned int cycle = 0; cycle < 8; ++cycle)
      {
        // pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, -1, 1);
            triangulation.refine_global(3);
          }
        else
          refine_grid();

        // pcout << "   Number of active cells:       "
        //       << triangulation.n_active_cells() << std::endl;

        setup_system();

        // pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        //       << " (by partition:";
        // for (unsigned int p = 0; p < n_mpi_processes; ++p)
        //   pcout << (p == 0 ? ' ' : '+')
        //         << (DoFTools::count_dofs_with_subdomain_association(dof_handler,
        //                                                             p));
        // pcout << ')' << std::endl;

        assemble_system();
        const unsigned int n_iterations = solve(file);

        pcout << "Solver converged in " << n_iterations << " iterations."
              << std::endl;

        // output_results(cycle);
      }
  }


  class MaterialProperties
  {
    public:
      MaterialProperties(double E, double nu)
        : E(E), nu(nu),
          mu(E / (2.0 * (1.0 + nu))),
          lambda((E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu)))
      {
        Assert(mu > 0, ExcMessage("Shear modulus must be positive"));
        Assert(nu > -1.0 && nu < 0.5, ExcMessage("Poisson ratio must be in (-1, 0.5)"));
      }

      double get_lambda() const { return lambda; }
      double get_mu() const { return mu; }

    private:
      double E;
      double nu;
      double mu;
      double lambda;
  };


  void generate_dataset(std::ofstream &file, std::string train_flag)
  {
    
    // samples (4800 = 1 × 2 × 12 × 50 × 4)  Stokes train dataset
    std::vector<double> E_values;
    std::cout<<"train flag: "<<train_flag<<std::endl;
    if (train_flag == "train")
      E_values = {2.5, 2.5e2, 2.5e4, 2.5e6};
    else
      E_values = {2.5e7};

    const std::vector<double> nu_values = {0.25, 0.3, 0.35};
    
    const std::vector<double> theta_values = ElasticProblem<2>::linspace(0.02, 0.9, 50); 
    
    unsigned int sample_index = 0;
    
    // Traverse
    for (double E: E_values)
    {
      for (double nu : nu_values)
      {
        MaterialProperties material(E, nu);

        for (double theta : theta_values) 
        {         
          
            ElasticProblem<2> solver(material.get_lambda(), material.get_mu());
            solver.set_theta(theta);
            solver.run(file);
            sample_index += 8;
            
            // if (sample_index % 100 == 0) {
            std::cout << "Generated " << sample_index << "/4800 samples" << std::endl;
            // }
          
        }        
      }
    }           
  }

} // namespace AMGElastic
