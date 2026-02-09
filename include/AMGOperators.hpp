#ifndef AMGOPERATORS_HPP
#define AMGOPERATORS_HPP

#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <omp.h>

namespace AMGOperators
{
    
    // Sparse matrix in CSR (Compressed Sparse Row) format
    struct CSRMatrix
    {
        int n_rows;
        int n_cols;
        std::vector<double> values;
        std::vector<int> col_indices;
        std::vector<int> row_ptr;

        CSRMatrix() : n_rows(0), n_cols(0) {}

        CSRMatrix(int rows, int cols)
            : n_rows(rows), n_cols(cols), row_ptr(rows + 1, 0) {}

        int nnz() const { return values.size(); }
    };

    /**
     * Compute strength of connection matrix
     * Node i is strongly connected to j if: -a_ij >= theta * max_k(-a_ik)
    **/    
    CSRMatrix compute_strength_matrix(const CSRMatrix& A, double theta)
    {
        const int n = A.n_rows;
        CSRMatrix S(n, n);

        // Compute maximum off-diagonal for each row
        std::vector<double> max_offdiag(n, 0.0);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            double max_val = 0.0;
            for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx)
            {
                int j = A.col_indices[idx];
                if (i != j)  // Off-diagonal only
                {
                    max_val = std::max(max_val, -A.values[idx]);
                }
            }
            max_offdiag[i] = max_val;
        }

        // Determine strong connections
        std::vector<std::vector<int>> strong_cols(n);
        std::vector<std::vector<double>> strong_vals(n);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx)
            {
                int j = A.col_indices[idx];
                if (i != j)  // Off-diagonal only
                {
                    if (-A.values[idx] >= theta * max_offdiag[i])
                    {
                        strong_cols[i].push_back(j);
                        strong_vals[i].push_back(1.0);
                    }
                }
            }
        }

        // Build CSR format
        S.row_ptr[0] = 0;
        for (int i = 0; i < n; ++i)
        {
            S.row_ptr[i + 1] = S.row_ptr[i] + strong_cols[i].size();
            S.col_indices.insert(S.col_indices.end(),
                                strong_cols[i].begin(), strong_cols[i].end());
            S.values.insert(S.values.end(),
                           strong_vals[i].begin(), strong_vals[i].end());
        }

        return S;
    }

    // Classical Ruge-Stüben C/F splitting
    std::vector<int> classical_cf_splitting(const CSRMatrix& A, double theta)
    {
        const int n = A.n_rows;

        // Compute strength matrix
        CSRMatrix S = compute_strength_matrix(A, theta);

        // Initialize all as undecided
        std::vector<int> cf_markers(n, 0);  // 0=undecided, 1=coarse, -1=fine

        // Compute lambda (number of strong connections)
        std::vector<int> lambda(n, 0);
        for (int i = 0; i < n; ++i)
        {
            lambda[i] = S.row_ptr[i + 1] - S.row_ptr[i];
        }

        std::set<int> undecided;
        for (int i = 0; i < n; ++i)
            undecided.insert(i);

        std::set<int> coarse_nodes;
        std::set<int> fine_nodes;

        // Greedy coarsening
        while (!undecided.empty())
        {
            // Find node with maximum lambda
            int max_node = -1;
            int max_lambda = -1;
            for (int i : undecided)
            {
                if (lambda[i] > max_lambda)
                {
                    max_lambda = lambda[i];
                    max_node = i;
                }
            }

            if (max_node == -1) break;

            // Make it coarse
            coarse_nodes.insert(max_node);
            cf_markers[max_node] = 1;
            undecided.erase(max_node);

            // Find strongly connected neighbors
            std::vector<int> neighbors;
            for (int idx = S.row_ptr[max_node]; idx < S.row_ptr[max_node + 1]; ++idx)
            {
                neighbors.push_back(S.col_indices[idx]);
            }

            // Make neighbors fine
            for (int j : neighbors)
            {
                if (undecided.find(j) != undecided.end())
                {
                    fine_nodes.insert(j);
                    cf_markers[j] = -1;
                    undecided.erase(j);

                    // Update lambda for neighbors of j
                    for (int idx = S.row_ptr[j]; idx < S.row_ptr[j + 1]; ++idx)
                    {
                        int k = S.col_indices[idx];
                        if (undecided.find(k) != undecided.end())
                        {
                            lambda[k]++;
                        }
                    }
                }
            }
        }

        // Convert to binary: 1=coarse, 0=fine
        std::vector<int> result(n);
        for (int i = 0; i < n; ++i)
        {
            result[i] = (cf_markers[i] == 1) ? 1 : 0;
        }

        return result;
    }

    // Extract coarse node indices   
    std::vector<int> extract_coarse_nodes(const std::vector<int>& cf_markers)
    {
        std::vector<int> coarse_nodes;
        for (size_t i = 0; i < cf_markers.size(); ++i)
        {
            if (cf_markers[i] == 1)
                coarse_nodes.push_back(i);
        }
        return coarse_nodes;
    }

    // Compute baseline prolongation matrix using direct interpolation
    CSRMatrix compute_baseline_prolongation(const CSRMatrix& A,
                                           const std::vector<int>& cf_markers)
    {
        const int n = A.n_rows;
        std::vector<int> coarse_nodes = extract_coarse_nodes(cf_markers);
        const int n_coarse = coarse_nodes.size();

        // Map coarse node index to column in P
        std::map<int, int> coarse_to_col;
        for (size_t i = 0; i < coarse_nodes.size(); ++i)
        {
            coarse_to_col[coarse_nodes[i]] = i;
        }

        CSRMatrix P(n, n_coarse);

        std::vector<std::vector<int>> P_cols(n);
        std::vector<std::vector<double>> P_vals(n);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            if (cf_markers[i] == 1)  // Coarse node
            {
                // Identity: P[i, coarse_to_col[i]] = 1.0
                P_cols[i].push_back(coarse_to_col[i]);
                P_vals[i].push_back(1.0);
            }
            else  // Fine node
            {
                // Find strongly connected coarse neighbors
                std::vector<int> coarse_neighbors;
                std::vector<double> weights;
                double diagonal_val = 0.0;

                for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx)
                {
                    int j = A.col_indices[idx];
                    double val = A.values[idx];

                    if (i == j)
                    {
                        diagonal_val = val;
                    }
                    else if (cf_markers[j] == 1)  // Coarse neighbor
                    {
                        coarse_neighbors.push_back(j);
                        weights.push_back(-val);  // Off-diagonal is negative
                    }
                }

                if (coarse_neighbors.empty())
                {
                    // No coarse neighbors, assign to first coarse node
                    if (!coarse_nodes.empty())
                    {
                        P_cols[i].push_back(0);
                        P_vals[i].push_back(1.0);
                    }
                }
                else
                {
                    // Direct interpolation: normalize weights
                    double total_weight = 0.0;
                    for (double w : weights)
                        total_weight += w;

                    if (std::abs(total_weight) > 1e-12)
                    {
                        for (size_t k = 0; k < coarse_neighbors.size(); ++k)
                        {
                            int c = coarse_neighbors[k];
                            double w = weights[k] / total_weight;
                            P_cols[i].push_back(coarse_to_col[c]);
                            P_vals[i].push_back(w);
                        }
                    }
                }
            }
        }

        // Build CSR format
        P.row_ptr[0] = 0;
        for (int i = 0; i < n; ++i)
        {
            P.row_ptr[i + 1] = P.row_ptr[i] + P_cols[i].size();
            P.col_indices.insert(P.col_indices.end(),
                                P_cols[i].begin(), P_cols[i].end());
            P.values.insert(P.values.end(),
                           P_vals[i].begin(), P_vals[i].end());
        }

        return P;
    }

    // Compute Gauss-Seidel relaxation matrix
    struct RelaxationMatrices
    {
        std::vector<double> D_inv;  // Inverse of diagonal
        CSRMatrix L;                 // Lower triangular
        CSRMatrix U;                 // Upper triangular
    };

    
    // Compute Gauss-Seidel smoother iteration matrix
    CSRMatrix compute_gauss_seidel_smoother(const CSRMatrix& A)
    {
        const int n = A.n_rows;
        CSRMatrix S(n, n);

        // For simplicity, return identity matrix
        // In practice, GS smoother is applied iteratively, not as a matrix
        S.row_ptr[0] = 0;
        for (int i = 0; i < n; ++i)
        {
            S.row_ptr[i + 1] = S.row_ptr[i] + 1;
            S.col_indices.push_back(i);
            S.values.push_back(1.0);
        }

        return S;
    }

} // namespace AMGOperators

#endif // AMGOPERATORS_HPP
