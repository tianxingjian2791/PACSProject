#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>


#include <deal.II/lac/generic_linear_algebra.h>

/* #define FORCE_USE_OF_TRILINOS */

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <cmath>
#include <fstream>
#include <iostream>

namespace AMGStokes
{
  using namespace dealii;



  namespace LinearSolvers
  {
    template <class Matrix, class Preconditioner>
    class InverseMatrix : public Subscriptor
    {
    public:
      InverseMatrix(const Matrix &m, const Preconditioner &preconditioner);

      template <typename VectorType>
      void vmult(VectorType &dst, const VectorType &src) const;

    private:
      const SmartPointer<const Matrix> matrix;
      const Preconditioner &           preconditioner;
    };


    template <class Matrix, class Preconditioner>
    InverseMatrix<Matrix, Preconditioner>::InverseMatrix(
      const Matrix &        m,
      const Preconditioner &preconditioner)
      : matrix(&m)
      , preconditioner(preconditioner)
    {}



    template <class Matrix, class Preconditioner>
    template <typename VectorType>
    void
    InverseMatrix<Matrix, Preconditioner>::vmult(VectorType &      dst,
                                                const VectorType &src) const
    {
      SolverControl        solver_control(src.size(), 1e-8 * src.l2_norm());
      SolverCG<VectorType> cg(solver_control);
      dst = 0;

      try
        {
          cg.solve(*matrix, dst, src, preconditioner);
        }
      catch (std::exception &e)
        {
          Assert(false, ExcMessage(e.what()));
        }
    }


    template <class PreconditionerA, class PreconditionerS>
    class BlockDiagonalPreconditioner : public Subscriptor
    {
    public:
      BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                  const PreconditionerS &preconditioner_S);

      void vmult(LA::MPI::BlockVector &      dst,
                const LA::MPI::BlockVector &src) const;

    private:
      const PreconditionerA &preconditioner_A;
      const PreconditionerS &preconditioner_S;
    };

    template <class PreconditionerA, class PreconditionerS>
    BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::
      BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                  const PreconditionerS &preconditioner_S)
      : preconditioner_A(preconditioner_A)
      , preconditioner_S(preconditioner_S)
    {}


    template <class PreconditionerA, class PreconditionerS>
    void BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::vmult(
      LA::MPI::BlockVector &      dst,
      const LA::MPI::BlockVector &src) const
    {
      preconditioner_A.vmult(dst.block(0), src.block(0));
      preconditioner_S.vmult(dst.block(1), src.block(1));
    }

  } // namespace LinearSolvers



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>(dim + 1)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
  };


  template <int dim>
  void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  values) const
  {
    const double R_x = p[0];
    const double R_y = p[1];

    constexpr double pi  = numbers::PI;
    constexpr double pi2 = numbers::PI * numbers::PI;

    values[0] = -1.0L / 2.0L * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0) *
                  std::exp(R_x * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0)) -
                0.4 * pi2 * std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::cos(2 * R_y * pi) +
                0.1 * std::pow(-std::sqrt(25.0 + 4 * pi2) + 5.0, 2) *
                  std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::cos(2 * R_y * pi);
    values[1] = 0.2 * pi * (-std::sqrt(25.0 + 4 * pi2) + 5.0) *
                  std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::sin(2 * R_y * pi) -
                0.05 * std::pow(-std::sqrt(25.0 + 4 * pi2) + 5.0, 3) *
                  std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::sin(2 * R_y * pi) / pi;
    values[2] = 0;
  }


  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(unsigned int bound_choice)
      : Function<dim>(dim + 1)
    {boundary_choice = bound_choice;}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override;
  private:
      unsigned int boundary_choice;
  };

  template <int dim>
  void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  values) const
  {
    values[0] = 0;
    values[1] = 0;
    values[2] = 0;

    if (boundary_choice == 0)
    {
      const double R_x = p[0];
      const double R_y = p[1];
  
      constexpr double pi  = numbers::PI;
      constexpr double pi2 = numbers::PI * numbers::PI;
  
      values[0] = -std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                    std::cos(2 * R_y * pi) +
                  1;
      values[1] = (1.0L / 2.0L) * (-std::sqrt(25.0 + 4 * pi2) + 5.0) *
                  std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::sin(2 * R_y * pi) / pi;
      values[2] =
        -1.0L / 2.0L * std::exp(R_x * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0)) -
        2.0 *
          (-6538034.74494422 +
          0.0134758939981709 * std::exp(4 * std::sqrt(25.0 + 4 * pi2))) /
          (-80.0 * std::exp(3 * std::sqrt(25.0 + 4 * pi2)) +
          16.0 * std::sqrt(25.0 + 4 * pi2) *
            std::exp(3 * std::sqrt(25.0 + 4 * pi2))) -
        1634508.68623606 * std::exp(-3.0 * std::sqrt(25.0 + 4 * pi2)) /
          (-10.0 + 2.0 * std::sqrt(25.0 + 4 * pi2)) +
        (-0.00673794699908547 * std::exp(std::sqrt(25.0 + 4 * pi2)) +
        3269017.37247211 * std::exp(-3 * std::sqrt(25.0 + 4 * pi2))) /
          (-8 * std::sqrt(25.0 + 4 * pi2) + 40.0) +
        0.00336897349954273 * std::exp(1.0 * std::sqrt(25.0 + 4 * pi2)) /
          (-10.0 + 2.0 * std::sqrt(25.0 + 4 * pi2));
    }
  }



  template <int dim>
  class StokesProblem
  {
  public:
    StokesProblem(unsigned int velocity_degree, double viscosity, unsigned int boundary_choice);

    void run(std::ofstream &file);
    void set_theta(double theta);
    void set_init_refinement(unsigned int refinement);
    void set_n_cycle(unsigned int n_cycle);

    // Getter methods for unified interface
    AMGOperators::CSRMatrix get_system_matrix_csr();
    double get_mesh_size() const;
    double get_convergence_factor() const;

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
    void refine_grid();
    // void output_results(const unsigned int cycle) const;
    void write_matrix_to_csv(const LA::MPI::BlockSparseMatrix &matrix, std::ofstream &file, double rho, double h);
    AMGOperators::CSRMatrix petsc_to_csr(const LA::MPI::BlockSparseMatrix &matrix);

    unsigned int velocity_degree;
    double       viscosity;
    double       theta;
    unsigned int init_refinement = 2;
    unsigned int n_cycle = 4;
    unsigned int boundary_choice;

    // Convergence metrics (stored after solve)
    double convergence_factor;
    double mesh_size_h;

    MPI_Comm     mpi_communicator;

    FESystem<dim>                             fe;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;

    std::vector<IndexSet> owned_partitioning;
    std::vector<IndexSet> relevant_partitioning;

    AffineConstraints<double> constraints;

    LA::MPI::BlockSparseMatrix system_matrix;
    LA::MPI::BlockSparseMatrix preconditioner_matrix;
    LA::MPI::BlockVector       locally_relevant_solution;
    LA::MPI::BlockVector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };



  template <int dim>
  StokesProblem<dim>::StokesProblem(unsigned int velocity_degree, double viscosity, unsigned int boundary_choice)
    : velocity_degree(velocity_degree)
    , viscosity(viscosity)
    , boundary_choice(boundary_choice)
    , mpi_communicator(MPI_COMM_WORLD)
    , fe(FE_Q<dim>(velocity_degree), dim, FE_Q<dim>(velocity_degree - 1), 1)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
  {}

  template <int dim>
  void StokesProblem<dim>::set_theta(double theta)
  {
    this -> theta = theta;
  }

  template <int dim>
  void StokesProblem<dim>::set_init_refinement(unsigned int refinement)
  {
    this -> init_refinement = refinement;
  }

  template <int dim>
  void StokesProblem<dim>::set_n_cycle(unsigned int n_cycle)
  {
    this -> n_cycle = n_cycle;
  }


  template <int dim>
  void StokesProblem<dim>::make_grid()
  {
    if (boundary_choice == 0)
    {
      GridGenerator::hyper_cube(triangulation, -0.5, 1.5);
    }
    else // The Lid-Driven Cavity Problem
    {
      // Generate the standard grids of the problem [0,1]^dim
      GridGenerator::hyper_cube(triangulation, 0, 1);
      
      // Mark the boundaries(y=1)
      for (auto &cell : triangulation.active_cell_iterators()) {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
          if (cell->face(f)->at_boundary()) {
            const Point<dim> center = cell->face(f)->center();
            // Check the top boundary(y≈1)
            if (std::abs(center[1] - 1.0) < 1e-5) {
              cell->face(f)->set_boundary_id(1); // top doundary
            } else {
              cell->face(f)->set_boundary_id(0); // side boundaries
            }
          }
        }
      }
    }
    triangulation.refine_global(init_refinement);  // old = 3, new = 2. Consider the computing ability of my laptop.
  }

  template <int dim>
  void StokesProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
    stokes_sub_blocks[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, stokes_sub_blocks);

    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, stokes_sub_blocks);

    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    // pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
    //       << n_u << '+' << n_p << ')' << std::endl;

    owned_partitioning.resize(2);
    owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_u);
    owned_partitioning[1] =
      dof_handler.locally_owned_dofs().get_view(n_u, n_u + n_p);

    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    {
      constraints.reinit(locally_relevant_dofs);
      const FEValuesExtractors::Vector velocities(0);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      
      if (boundary_choice == 0) // Typical Stokes problem
      {
        VectorTools::interpolate_boundary_values(dof_handler,
          0,
          ExactSolution<dim>(boundary_choice),
          constraints,
          fe.component_mask(velocities));
      }
      else // The Lid-Driven Cavity Problem
      {
        // Side boundaries(ID=0): 0 velocity
        VectorTools::interpolate_boundary_values(
          dof_handler,
          0,
          Functions::ZeroFunction<dim>(dim+1),
          constraints,
          fe.component_mask(velocities));
        
        // Top boundary(ID=1): The velocity on x is 1，on y is 0
        Vector<double> lid_velocity_values(dim+1);
        for (auto &v: lid_velocity_values)
          v = 0.0;
        lid_velocity_values[0] = 1.0; // x-velocity = 1
        Functions::ConstantFunction<dim> lid_velocity(lid_velocity_values);
        
        VectorTools::interpolate_boundary_values(
          dof_handler,
          1,
          lid_velocity,
          constraints,
          fe.component_mask(velocities));
      }
      constraints.close();
    }

    {
      system_matrix.clear();

      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (c == dim && d == dim)
            coupling[c][d] = DoFTools::none;
          else if (c == dim || d == dim || c == d)
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      BlockDynamicSparsityPattern dsp(relevant_partitioning);

      DoFTools::make_sparsity_pattern(
        dof_handler, coupling, dsp, constraints, false);

      SparsityTools::distribute_sparsity_pattern(
        dsp,
        dof_handler.locally_owned_dofs(),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    }

    {
      preconditioner_matrix.clear();

      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (c == dim && d == dim)
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      BlockDynamicSparsityPattern dsp(relevant_partitioning);

      DoFTools::make_sparsity_pattern(
        dof_handler, coupling, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        Utilities::MPI::all_gather(mpi_communicator,
                                  dof_handler.locally_owned_dofs()),
        mpi_communicator,
        locally_relevant_dofs);
      preconditioner_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    }

    locally_relevant_solution.reinit(owned_partitioning,
                                    relevant_partitioning,
                                    mpi_communicator);
    system_rhs.reinit(owned_partitioning, mpi_communicator);
  }



  template <int dim>
  void StokesProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix         = 0;
    preconditioner_matrix = 0;
    system_rhs            = 0;

    const QGauss<dim> quadrature_formula(velocity_degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix2(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    const RightHandSide<dim>    right_hand_side;
    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector     velocities(0);
    const FEValuesExtractors::Scalar     pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix  = 0;
          cell_matrix2 = 0;
          cell_rhs     = 0;

          fe_values.reinit(cell);
          // right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
          //                                   rhs_values);

          // Modify right handside：set 0 when it's the Lid-Driven Cavity Problem
          if (boundary_choice == 0) {
            right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                              rhs_values);
          } else { // Driven problem
            for (auto &vec : rhs_values) {
              vec = 0; // 0 source item
            }
          }
          
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                  div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                  phi_p[k]      = fe_values[pressure].value(k, q);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) +=
                        (viscosity *
                          scalar_product(grad_phi_u[i], grad_phi_u[j]) -
                        div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                        fe_values.JxW(q);

                      cell_matrix2(i, j) += 1.0 / viscosity * phi_p[i] *
                                            phi_p[j] * fe_values.JxW(q);
                    }

                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                  cell_rhs(i) += fe_values.shape_value(i, q) *
                                rhs_values[q](component_i) * fe_values.JxW(q);
                }
            }


          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);

          constraints.distribute_local_to_global(cell_matrix2,
                                                local_dof_indices,
                                                preconditioner_matrix);
        }

    system_matrix.compress(VectorOperation::add);
    preconditioner_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }



  template <int dim>
  void StokesProblem<dim>::solve(std::ofstream &file)
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::PreconditionAMG prec_A;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
      data.symmetric_operator = true;
      data.strong_threshold = theta;
#endif
      prec_A.initialize(system_matrix.block(0, 0), data);
    }

    LA::MPI::PreconditionAMG prec_S;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
      data.symmetric_operator = true;
#endif
      prec_S.initialize(preconditioner_matrix.block(1, 1), data);
    }

    using mp_inverse_t = LinearSolvers::InverseMatrix<LA::MPI::SparseMatrix,
                                                      LA::MPI::PreconditionAMG>;
    const mp_inverse_t mp_inverse(preconditioner_matrix.block(1, 1), prec_S);

    const LinearSolvers::BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG,
                                                    mp_inverse_t>
      preconditioner(prec_A, mp_inverse);

    SolverControl solver_control(system_matrix.m(), 1e-10 * system_rhs.l2_norm());

    SolverMinRes<LA::MPI::BlockVector> solver(solver_control);

    LA::MPI::BlockVector distributed_solution(owned_partitioning,
                                              mpi_communicator);

    constraints.set_zero(distributed_solution);
    LA::MPI::BlockVector residual(system_rhs);

    system_matrix.vmult(residual, distributed_solution);
    residual -= system_rhs;
    double init_r_norm = residual.l2_norm();
    // std::cout<<init_r_norm<<std::endl;


    solver.solve(system_matrix,
                distributed_solution,
                system_rhs,
                preconditioner);
    
    system_matrix.vmult(residual, distributed_solution);
    residual -= system_rhs;
    double final_r_norm = residual.l2_norm();  
    // std::cout<<final_r_norm<<std::endl;

    const unsigned int k = solver_control.last_step();
    if (k < 1) 
    {
      std::cerr << "Warning: Insufficient residuals recorded (" 
                << k << "). Returning rho=0." << std::endl;
      return;
    }

    // ρ = (||r_k|| / ||r_0||)^{1/k}
    const double rho = (k > 0) ? std::pow(final_r_norm / init_r_norm, 1.0 / k) : 0.0;
    double h = triangulation.begin_active()->diameter(); // The size of grids

    // Store convergence metrics for getter methods
    convergence_factor = rho;
    mesh_size_h = h;

    write_matrix_to_csv(system_matrix, file, rho, h);

    // pcout << "   Solved in " << solver_control.last_step() << " iterations."
    //       << std::endl;

    constraints.distribute(distributed_solution);

    locally_relevant_solution = distributed_solution;
    if (boundary_choice == 0)
    {
      const double mean_pressure =
      VectorTools::compute_mean_value(dof_handler,
                                      QGauss<dim>(velocity_degree + 2),
                                      locally_relevant_solution,
                                      dim);
      distributed_solution.block(1).add(-mean_pressure);
      locally_relevant_solution.block(1) = distributed_solution.block(1);      
    }
  }



  template <int dim>
  void StokesProblem<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    triangulation.refine_global();
  }


  /*
  template <int dim>
  void StokesProblem<dim>::output_results(const unsigned int cycle) const
  {
    {
      const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
      const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                      dim + 1);

      Vector<double> cellwise_errors(triangulation.n_active_cells());
      QGauss<dim>    quadrature(velocity_degree + 2);

      VectorTools::integrate_difference(dof_handler,
                                        locally_relevant_solution,
                                        ExactSolution<dim>(),
                                        cellwise_errors,
                                        quadrature,
                                        VectorTools::L2_norm,
                                        &velocity_mask);

      const double error_u_l2 =
        VectorTools::compute_global_error(triangulation,
                                          cellwise_errors,
                                          VectorTools::L2_norm);

      VectorTools::integrate_difference(dof_handler,
                                        locally_relevant_solution,
                                        ExactSolution<dim>(),
                                        cellwise_errors,
                                        quadrature,
                                        VectorTools::L2_norm,
                                        &pressure_mask);

      const double error_p_l2 =
        VectorTools::compute_global_error(triangulation,
                                          cellwise_errors,
                                          VectorTools::L2_norm);

      pcout << "error: u_0: " << error_u_l2 << " p_0: " << error_p_l2
            << std::endl;
    }


    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                            solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);

    LA::MPI::BlockVector interpolated;
    interpolated.reinit(owned_partitioning, MPI_COMM_WORLD);
    VectorTools::interpolate(dof_handler, ExactSolution<dim>(), interpolated);

    LA::MPI::BlockVector interpolated_relevant(owned_partitioning,
                                              relevant_partitioning,
                                              MPI_COMM_WORLD);
    interpolated_relevant = interpolated;
    {
      std::vector<std::string> solution_names(dim, "ref_u");
      solution_names.emplace_back("ref_p");
      data_out.add_data_vector(interpolated_relevant,
                              solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    }


    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2);
  }
  */

  template <int dim>
  void StokesProblem<dim>::write_matrix_to_csv(const LA::MPI::BlockSparseMatrix &matrix,
    std::ofstream &file,
    double rho,
    double h)
  {  
    // Just access to (0,0) block（velocity-velocity submatrix）
    const auto &petsc_matrix= matrix.block(0,0);
    
    // const Mat petsc_matrix = A00.petsc_matrix();
    
    PetscInt m, n;
    MatGetSize(petsc_matrix, &m, &n);
    
    PetscInt start, end;
    MatGetOwnershipRange(petsc_matrix, &start, &end);
    
    std::vector<PetscInt> row_ptr;
    std::vector<PetscInt> col_ind;
    std::vector<PetscScalar> values;
    
    PetscInt idx = 0;
    for (PetscInt i = start; i < end; i++) {
        PetscInt ncols;
        const PetscInt* cols;
        const PetscScalar* vals;
        
        MatGetRow(petsc_matrix, i, &ncols, &cols, &vals);
        
        row_ptr.push_back(idx);
        const double zero_tol = 1e-12;
        
        for (PetscInt j = 0; j < ncols; j++) {
            if (std::abs(vals[j]) > zero_tol) {
                col_ind.push_back(cols[j]);
                values.push_back(vals[j]);
                idx++;
            }
        }
        MatRestoreRow(petsc_matrix, i, &ncols, &cols, &vals);
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
  void StokesProblem<dim>::run(std::ofstream &file)
  {
#ifdef USE_PETSC_LA
    // pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif
    const unsigned int n_cycles = n_cycle;  // {2, 3, 4, 5} as the train dataset, {6} as the test dataset.
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        // pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          make_grid();
        else
          refine_grid();

        setup_system();

        assemble_system();
        solve(file);

        // if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
        //   {
        //     TimerOutput::Scope t(computing_timer, "output");
        //     output_results(cycle);
        //   }

        // computing_timer.print_summary();
        // computing_timer.reset();

        pcout << std::endl;
      }
  }


  // ============================================================================
  // Getter Method Implementations
  // ============================================================================

  template <int dim>
  AMGOperators::CSRMatrix StokesProblem<dim>::get_system_matrix_csr()
  {
    return petsc_to_csr(system_matrix);
  }

  template <int dim>
  double StokesProblem<dim>::get_mesh_size() const
  {
    return mesh_size_h;
  }

  template <int dim>
  double StokesProblem<dim>::get_convergence_factor() const
  {
    return convergence_factor;
  }

  // ============================================================================
  // PETSc Block Matrix to CSR Conversion
  // ============================================================================

  template <int dim>
  AMGOperators::CSRMatrix StokesProblem<dim>::petsc_to_csr(const LA::MPI::BlockSparseMatrix &matrix)
  {
    // For Stokes, we extract the velocity block (0,0) as the main system matrix
    // This is the block that uses AMG preconditioner
    const auto &A_block = matrix.block(0, 0);

    const unsigned int m = A_block.m();
    const unsigned int n = A_block.n();

    PetscInt start, end;
    MatGetOwnershipRange(A_block, &start, &end);

    std::vector<PetscInt> row_ptr;
    std::vector<PetscInt> col_ind;
    std::vector<PetscScalar> values;

    PetscInt idx = 0;
    for (PetscInt i = start; i < end; i++) {
      PetscInt ncols;
      const PetscInt* cols;
      const PetscScalar* vals;
      MatGetRow(A_block, i, &ncols, &cols, &vals);

      row_ptr.push_back(idx);
      double zero_tol = 1e-12;
      for (PetscInt j = 0; j < ncols; j++) {
        if (std::abs(vals[j]) > zero_tol) {
          col_ind.push_back(cols[j]);
          values.push_back(vals[j]);
          idx++;
        }
      }
      MatRestoreRow(A_block, i, &ncols, &cols, &vals);
    }
    row_ptr.push_back(idx);

    // Build CSRMatrix
    AMGOperators::CSRMatrix A;
    A.n_rows = m;
    A.n_cols = n;
    A.row_ptr.resize(row_ptr.size());
    A.col_indices.resize(col_ind.size());
    A.values.resize(values.size());

    for (size_t i = 0; i < row_ptr.size(); ++i)
      A.row_ptr[i] = row_ptr[i];
    for (size_t i = 0; i < col_ind.size(); ++i)
      A.col_indices[i] = col_ind[i];
    for (size_t i = 0; i < values.size(); ++i)
      A.values[i] = values[i];

    return A;
  }

  // ============================================================================
  // DEPRECATED: Legacy dataset generation (use generate_amg_data instead)
  // ============================================================================

  void generate_dataset(std::ofstream &file, std::string train_flag)
  {
    std::cerr << "ERROR: generate_dataset() is deprecated." << std::endl;
    std::cerr << "Please use the unified generator: ./generate_amg_data -p S -s train -f all -c medium" << std::endl;
    throw std::runtime_error("Deprecated function called. Use unified generator instead.");
  }

} // namespace AMGStokes