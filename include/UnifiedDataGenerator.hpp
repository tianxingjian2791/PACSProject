#ifndef UNIFIEDDATAGENERATOR_HPP
#define UNIFIEDDATAGENERATOR_HPP

#include "AMGOperators.hpp"
#include "Pooling.hpp"
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <omp.h>
#include <mpi.h>

namespace UnifiedAMG
{
    using namespace dealii;
    using namespace AMGOperators;

    /**
     * Output format type
     */
    enum class OutputFormat
    {
        THETA_CNN,    // CNN format: pooled 50x50 matrices
        THETA_GNN,    // GNN format: sparse graph for theta prediction
        P_VALUE,      // P-value format: graph with C/F splitting and P
        ALL           // Generate all formats
    };

    /**
     * Convert deal.II PETSc sparse matrix to CSR format
     */
    CSRMatrix convert_to_csr(const PETScWrappers::MPI::SparseMatrix& matrix)
    {
        CSRMatrix A;
        A.n_rows = matrix.m();
        A.n_cols = matrix.n();

        // Reserve space
        A.row_ptr.resize(A.n_rows + 1, 0);

        // Extract non-zero values
        for (unsigned int i = 0; i < A.n_rows; ++i)
        {
            std::vector<PetscInt> cols;
            std::vector<PetscScalar> vals;

            // Get row data
            PetscInt ncols;
            const PetscInt* col_indices;
            const PetscScalar* values;

            // This is a simplified version - actual implementation needs proper PETSc calls
            // For now, iterate through columns
            for (unsigned int j = 0; j < A.n_cols; ++j)
            {
                if (matrix.el(i, j) != 0.0)
                {
                    A.col_indices.push_back(j);
                    A.values.push_back(matrix.el(i, j));
                }
            }

            A.row_ptr[i + 1] = A.col_indices.size();
        }

        return A;
    }

    /**
     * Find optimal theta by searching over candidates
     *
     * @param A System matrix
     * @param theta_candidates Candidate theta values to test
     * @param rho_out Output convergence factor
     * @return Optimal theta value
     */
    double find_optimal_theta(const CSRMatrix& A,
                              const std::vector<double>& theta_candidates,
                              double& rho_out)
    {
        double best_theta = theta_candidates[0];
        double best_rho = 1.0;

        // Parallel search over theta candidates
        #pragma omp parallel
        {
            double local_best_theta = theta_candidates[0];
            double local_best_rho = 1.0;

            #pragma omp for
            for (size_t i = 0; i < theta_candidates.size(); ++i)
            {
                double theta = theta_candidates[i];

                // Perform C/F splitting
                std::vector<int> cf_markers = classical_cf_splitting(A, theta);

                // Count coarse nodes
                int n_coarse = 0;
                for (int marker : cf_markers)
                    n_coarse += marker;

                // Heuristic: estimate convergence factor
                // In practice, would need to compute actual two-grid convergence
                // For now, use coarsening ratio as proxy
                double coarsening_ratio = static_cast<double>(n_coarse) / A.n_rows;

                // Ideal coarsening ratio is around 0.25-0.5
                double rho_estimate = std::abs(coarsening_ratio - 0.35) + 0.1;

                if (rho_estimate < local_best_rho)
                {
                    local_best_rho = rho_estimate;
                    local_best_theta = theta;
                }
            }

            // Update global best (critical section)
            #pragma omp critical
            {
                if (local_best_rho < best_rho)
                {
                    best_rho = local_best_rho;
                    best_theta = local_best_theta;
                }
            }
        }

        rho_out = best_rho;
        return best_theta;
    }

    /**
     * Write theta CNN format (pooled 50x50 matrix)
     */
    void write_theta_cnn_format(std::ofstream& file,
                                const CSRMatrix& A,
                                double theta,
                                double rho,
                                double h,
                                int pool_size = 50)
    {
        // Pool matrix to 50x50
        std::vector<std::vector<double>> V;
        std::vector<std::vector<int>> C;

        parallel_pooling_csr(A.values, A.col_indices, A.row_ptr,
                            A.n_rows, pool_size, PoolingOp::SUM, V, C);

        // Standardize
        std_normalize(V);

        // Write: rho, theta, log_h, [2500 pooled values]
        double log_h = -std::log2(h);
        file << rho << "," << theta << "," << log_h;

        for (const auto& row : V)
        {
            for (double val : row)
            {
                file << "," << val;
            }
        }
        file << "\n";
    }

    /**
     * Write theta GNN format (sparse graph)
     */
    void write_theta_gnn_format(std::ofstream& file,
                                const CSRMatrix& A,
                                double theta,
                                double rho,
                                double h)
    {
        // Write: num_rows, num_cols, theta, rho, h, nnz,
        //        [values], [row_ptr], [col_indices]
        file << A.n_rows << "," << A.n_cols << "," << theta << ","
             << rho << "," << h << "," << A.nnz();

        // Write values
        for (double val : A.values)
            file << "," << val;

        // Write row_ptr
        for (int ptr : A.row_ptr)
            file << "," << ptr;

        // Write col_indices
        for (int col : A.col_indices)
            file << "," << col;

        file << "\n";
    }

    /**
     * Write P-value format (graph with C/F splitting and baseline P)
     */
    void write_p_value_format(std::ofstream& file,
                             const CSRMatrix& A,
                             double theta,
                             double rho,
                             double h,
                             const std::vector<int>& cf_markers,
                             const CSRMatrix& P,
                             const CSRMatrix& S)
    {
        // Write matrix A
        file << A.n_rows << "," << A.n_cols << "," << theta << ","
             << rho << "," << h << "," << A.nnz();

        for (double val : A.values)
            file << "," << val;
        for (int ptr : A.row_ptr)
            file << "," << ptr;
        for (int col : A.col_indices)
            file << "," << col;

        // Write coarse nodes
        std::vector<int> coarse_nodes = extract_coarse_nodes(cf_markers);
        file << "," << coarse_nodes.size();
        for (int node : coarse_nodes)
            file << "," << node;

        // Write prolongation matrix P
        file << "," << P.nnz();
        for (double val : P.values)
            file << "," << val;
        for (int ptr : P.row_ptr)
            file << "," << ptr;
        for (int col : P.col_indices)
            file << "," << col;

        // Write smoother matrix S
        file << "," << S.nnz();
        for (double val : S.values)
            file << "," << val;
        for (int ptr : S.row_ptr)
            file << "," << ptr;
        for (int col : S.col_indices)
            file << "," << col;

        file << "\n";
    }

    /**
     * Generate unified dataset from a PETSc sparse matrix
     *
     * @param matrix System matrix from deal.II
     * @param h Mesh size
     * @param output_format Which format(s) to generate
     * @param theta_cnn_file Output file for CNN format (optional)
     * @param theta_gnn_file Output file for GNN format (optional)
     * @param p_value_file Output file for P-value format (optional)
     */
    void generate_unified_sample(const PETScWrappers::MPI::SparseMatrix& matrix,
                                 double h,
                                 OutputFormat output_format,
                                 std::ofstream* theta_cnn_file = nullptr,
                                 std::ofstream* theta_gnn_file = nullptr,
                                 std::ofstream* p_value_file = nullptr)
    {
        // Convert to CSR format
        CSRMatrix A = convert_to_csr(matrix);

        // Find optimal theta
        std::vector<double> theta_candidates;
        for (int i = 0; i <= 20; ++i)
        {
            theta_candidates.push_back(0.05 + i * 0.025);  // 0.05 to 0.6
        }

        double rho;
        double optimal_theta = find_optimal_theta(A, theta_candidates, rho);

        // Compute C/F splitting with optimal theta
        std::vector<int> cf_markers = classical_cf_splitting(A, optimal_theta);

        // Compute baseline prolongation
        CSRMatrix P = compute_baseline_prolongation(A, cf_markers);

        // Compute smoother (simplified - identity for now)
        CSRMatrix S = compute_gauss_seidel_smoother(A);

        // Write outputs based on format
        if ((output_format == OutputFormat::THETA_CNN || output_format == OutputFormat::ALL)
            && theta_cnn_file != nullptr)
        {
            write_theta_cnn_format(*theta_cnn_file, A, optimal_theta, rho, h);
        }

        if ((output_format == OutputFormat::THETA_GNN || output_format == OutputFormat::ALL)
            && theta_gnn_file != nullptr)
        {
            write_theta_gnn_format(*theta_gnn_file, A, optimal_theta, rho, h);
        }

        if ((output_format == OutputFormat::P_VALUE || output_format == OutputFormat::ALL)
            && p_value_file != nullptr)
        {
            write_p_value_format(*p_value_file, A, optimal_theta, rho, h,
                                cf_markers, P, S);
        }
    }

} // namespace UnifiedAMG

#endif // UNIFIEDDATAGENERATOR_HPP
