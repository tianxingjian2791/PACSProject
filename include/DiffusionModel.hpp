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
#include <deal.II/lac/solver_control.h> 
#include <deal.II/lac/petsc_sparse_matrix.h> 
#include <deal.II/lac/petsc_vector.h> 
#include <deal.II/lac/petsc_solver.h> 
#include <deal.II/lac/petsc_precondition.h> 
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

#include <fstream>
#include <iostream>
#include <cmath>


namespace AMGDiffusion
{
  using namespace dealii;

  // Define 4 modes
  enum class DiffusionPattern
  {
    vertical_stripes,    // (a) 
    vertical_stripes2,  // (b)
    checkerboard2,        // (c)
    checkerboard       // (d)
  };

  // The function of exact solution（Choose according to the mode）
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

  // THe right handside function（Choose by the mode）
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

  // The function for different diffusion coefficients
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
      case DiffusionPattern::vertical_stripes: //(a)
      {
        // Divide the area in to 2 stripes: (-1, -0.0), [, 1)
        if (x < -0.0 + tol)
          return 1.0; // gray stripe
        else
          return std::pow(10.0, epsilon); // white stripe
      }
        
      case DiffusionPattern::vertical_stripes2: // (b)
      {
        // Divide the area in to 4 stripes: (-1, -0.5), [-0.5, 0), [0, 0.5), [0.5, 1)
        if (x < -0.5 + tol)
          return 1.0;  // gray
        else if (x < 0.0 + tol)
          return std::pow(10.0, epsilon);  // white
        else if (x < 0.5 + tol)
          return 1.0;  // gray
        else
          return std::pow(10.0, epsilon);  // white
      }

      case DiffusionPattern::checkerboard2: // 4 x 4 checkerboard (c)
      {
        int i = static_cast<int>(std::floor((x + 1.0) / 0.5));
        int j = static_cast<int>(std::floor((y + 1.0) / 0.5));
        i = std::min(i, 3);
        j = std::min(j, 3);

        if ((i + j) % 2 == 0)
          return 1.0;  // gray
        else
          return std::pow(10.0, epsilon);  //white
      }

      case DiffusionPattern::checkerboard: // 2 x 2 checkerboard (d)
      {
        int i = static_cast<int>(std::floor(x + 1.0));
        int j = static_cast<int>(std::floor(y + 1.0));
        i = std::min(i, 1);
        j = std::min(j, 1);

        if ((i + j) % 2 == 0)
          return 1.0;  // gray
        else
          return std::pow(10.0, epsilon);  // white
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

  // Solver
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


    // support static function
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
    void write_matrix_to_csv(const PETScWrappers::MPI::SparseMatrix &matrix, std::ofstream &file, double rho, double h);


    // mode parameter
    DiffusionPattern pattern;
    double theta;
    double epsilon;
    unsigned int refinement;

    // Grids and finite elements 
    dealii::Triangulation<dim> triangulation;
    dealii::FE_Q<dim> fe;
    dealii::DoFHandler<dim> dof_handler;
    dealii::SolverControl solver_control; // solver controller

    // Constraint and system matrix
    dealii::AffineConstraints<double> constraints;
    dealii::PETScWrappers::MPI::SparseMatrix system_matrix;
    dealii::PETScWrappers::MPI::Vector solution;
    // dealii::PETScWrappers::MPI::Vector init_solution;
    dealii::PETScWrappers::MPI::Vector system_rhs;

    // Exact solution and right hand
    ExactSolution<dim> exact_solution;
    RightHandSide<dim> right_hand_side;
    DiffusionCoefficient<dim> diffusion_coefficient;
  };

  template <int dim>
  Solver<dim>::Solver(DiffusionPattern pattern, double epsilon, unsigned int refinement)
    : pattern(pattern)
    , epsilon(epsilon)
    , refinement(refinement)
    , fe(1) // Q1 FE
    , dof_handler(triangulation)
    , solver_control(1000, 1e-12) // max iterations is 1000，tolerance is 1e-12
    , exact_solution(pattern)
    , right_hand_side(pattern)
    , diffusion_coefficient(pattern, epsilon)
  {}

  template <int dim>
  void Solver<dim>::make_grid()
  {
    // Generate square grids 
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
    this->refinement = refinement;
  }

  template <int dim>
  void Solver<dim>::setup_system()
  {
    // Setup dofs
    dof_handler.distribute_dofs(fe);
    // std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    // Initialize MPI variables
    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Create sparsity pattern
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dsp.compress();

    // Initialize matrices and vectors
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    // init_solution = solution;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    // Setup boundary conditions
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

    // Traverse all the cells
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit(cell);

      // Get the diffusion coefficient of the current cell
      const double mu = diffusion_coefficient.value(fe_values.quadrature_point(0));

      // Assemble the matrix and right handside for the cell
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

          // right handside：f * phi_i
          cell_rhs(i) += (right_hand_side.value(fe_values.quadrature_point(q_index)) *
                          fe_values.shape_value(i, q_index) *
                          fe_values.JxW(q_index));
        }
      }

      // Add the cell contribution to the global system
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                            cell_rhs,
                                            local_dof_indices,
                                            system_matrix,
                                            system_rhs);
    }

    // apply the constraint to and comress the matrix
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void Solver<dim>::solve(std::ofstream &file)
  {
    // solution = init_solution;
    // setup the parameters of the solver
    dealii::PETScWrappers::SolverCG solver(solver_control, MPI_COMM_WORLD);
    dealii::PETScWrappers::PreconditionBoomerAMG preconditioner;

    // configure the parameters of BoomerAMG
    dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
    data.strong_threshold = theta;  // setup the strong threshold θ(adjustable)
    // std::cout<<"strong threshold: "<<data.strong_threshold<<std::endl;
    data.symmetric_operator = true; // setup the symmetric operator

    // Initialize the preconditioner of AMG
    preconditioner.initialize(system_matrix, data);

    PETScWrappers::MPI::Vector residual(system_rhs);

    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    double init_r_norm = residual.l2_norm();
    // std::cout<<init_r_norm<<" ";

    // Solve the system
    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    double final_r_norm = residual.l2_norm();  
    // std::cout<<final_r_norm<<std::endl;


    // Print the iterative information
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
    double h = triangulation.begin_active()->diameter(); // The size of grids
    write_matrix_to_csv(system_matrix, file, rho, h);
    

    // Apply the constraints
    constraints.distribute(solution);

    // return rho;
  }
    
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
  
    // write m(rows), n(cols), rho, h, nnz
    file << m << "," << n << "," << theta << "," << rho << "," << h << "," << values.size();
  
    // write non-zero value
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
  void Solver<dim>::run(std::ofstream &file)
  {
    make_grid();
    setup_system();
    assemble_system();
    
    solve(file);

  }

  void generate_dataset(std::ofstream &file, std::string train_flag)
  {
    //(4800 samples = 4 modes × 12ε × 25θ × 8 grids)
    const std::array<DiffusionPattern, 4> patterns = 
    {{
      DiffusionPattern::vertical_stripes,
      DiffusionPattern::vertical_stripes2,
      DiffusionPattern::checkerboard2,
      DiffusionPattern::checkerboard
    }};
  
    std::vector<double> epsilon_values;
    if (train_flag == "train") 
      // Solver<2>::linspace(0.0, 9.5, 40); // dataset2
      epsilon_values = Solver<2>::linspace(0.0, 9.5, 12); // dataset1
    else
      epsilon_values = {1.0, 2.0, 3.0};
    
    const std::vector<double> theta_values = 
        // Solver<2>::linspace(0.02, 0.9, 45); // dataset2
        Solver<2>::linspace(0.02, 0.9, 25); // dataset1

    std::vector<unsigned int> refinements = {3, 4, 5, 6};

    // if (train_flag == "train")
    //   refinements = {3, 4, 5, 6}; // dataset: train
    // else
    //   refinements = {7}; // dataset: test

    
    unsigned int sample_index = 0;
    
    // Traverse all the combinations of different parameters
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
