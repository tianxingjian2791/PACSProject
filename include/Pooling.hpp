#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <string>
#include <sstream>

enum class PoolingOp {
    SUM,
    MAX
};

// Parallel pooling function
void parallel_pooling_coo(const std::vector<double>& val,
                      const std::vector<int>& row,
                      const std::vector<int>& col,
                      int n, int m, PoolingOp op,
                      std::vector<std::vector<double>>& V,
                      std::vector<std::vector<int>>& C) {
    
    // Check input consistency
    if (val.size() != row.size() || val.size() != col.size()) {
        std::invalid_argument("val, row, col must have same size");
    }
    
    const int nnz = val.size();
    if (nnz == 0) {
        V = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
        C = std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
        return;
    }
    
    // Calculate block partitioning parameters
    const int q = n / m;          // Base block size
    const int p = n % m;          // Number of extra blocks
    const int t = (q + 1) * p;    // Boundary point
    
    // Initialize output matrix
    V = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
    C = std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
    
    // Method 1: Thread-private matrix merge (suitable for small m)
    if (m <= 100) {
        // Get number of threads
        int num_threads = omp_get_max_threads();
        
        // Thread-private matrices
        std::vector<std::vector<std::vector<double>>> V_private(
            num_threads, std::vector<std::vector<double>>(
                m, std::vector<double>(m, (op == PoolingOp::MAX) ? 
                    -std::numeric_limits<double>::max() : 0.0)));
        
        std::vector<std::vector<std::vector<int>>> C_private(
            num_threads, std::vector<std::vector<int>>(m, std::vector<int>(m, 0)));
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            
            #pragma omp for
            for (int k = 0; k < nnz; k++) {
                int i = row[k];
                int j = col[k];
                
                // Calculate pooling coordinates
                int I, J;
                if (i < t) {
                    I = i / (q + 1);
                } else {
                    I = (i - t) / q + p;
                }
                
                if (j < t) {
                    J = j / (q + 1);
                } else {
                    J = (j - t) / q + p;
                }
                
                // Ensure within valid range
                if (I >= 0 && I < m && J >= 0 && J < m) {
                    // Update based on operation type
                    if (op == PoolingOp::SUM) {
                        V_private[tid][I][J] += val[k];
                    } else { // MAX
                        if (val[k] > V_private[tid][I][J]) {
                            V_private[tid][I][J] = val[k];
                        }
                    }
                    C_private[tid][I][J]++;
                }
            }
        }
        
        // Merge thread-private matrices
        for (int tid = 0; tid < num_threads; tid++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    if (op == PoolingOp::SUM) {
                        V[i][j] += V_private[tid][i][j];
                    } else { // MAX
                        if (V_private[tid][i][j] > V[i][j]) {
                            V[i][j] = V_private[tid][i][j];
                        }
                    }
                    C[i][j] += C_private[tid][i][j];
                }
            }
        }
    }
    // Method 2: Atomic operations (suitable for large m)
    else {
        #pragma omp parallel for
        for (int k = 0; k < nnz; k++) {
            int i = row[k];
            int j = col[k];
            
            // Calculate pooling coordinates
            int I, J;
            if (i < t) {
                I = i / (q + 1);
            } else {
                I = (i - t) / q + p;
            }
            
            if (j < t) {
                J = j / (q + 1);
            } else {
                J = (j - t) / q + p;
            }
            
            // Ensure within valid range
            if (I >= 0 && I < m && J >= 0 && J < m) {
                // Use different synchronization strategies based on operation type
                if (op == PoolingOp::SUM) {
                    #pragma omp atomic
                    V[I][J] += val[k];
                    #pragma omp atomic
                    C[I][J]++;
                } else { // MAX
                    // Use critical section to protect max updates
                    #pragma omp critical
                    {
                        if (val[k] > V[I][J]) {
                            V[I][J] = val[k];
                        }
                        C[I][J]++;
                    }
                }
            }
        }
    }
}

// Parallel pooling function supporting CSR format
void parallel_pooling_csr(const std::vector<double>& values,
                          const std::vector<int>& col_indices,
                          const std::vector<int>& row_ptr,
                          int n, int m, PoolingOp op,
                          std::vector<std::vector<double>>& V,
                          std::vector<std::vector<int>>& C) 
{
    // Check input validity
    if (values.size() != col_indices.size()) {
        throw std::invalid_argument("values and col_indices must have same size");
    }
    
    const int nnz = values.size();
    if (nnz == 0) {
        V = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
        C = std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
        return;
    }
    
    // Check row_ptr validity
    if (row_ptr.empty() || row_ptr.back() != nnz) {
        throw std::invalid_argument("Invalid row_ptr array");
    }
    
    // Calculate block partitioning parameters
    const int q = n / m;          // Base block size
    const int p = n % m;          // Number of extra blocks
    const int t = (q + 1) * p;    // Boundary point
    
    // Initialize output matrix
    V = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
    C = std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
    
    #pragma omp parallel
    {
        // Thread-private matrices
        std::vector<std::vector<double>> V_local(m, std::vector<double>(m, 
            (op == PoolingOp::MAX) ? -std::numeric_limits<double>::max() : 0.0));
        std::vector<std::vector<int>> C_local(m, std::vector<int>(m, 0));
        
        // Process each row in parallel
        #pragma omp for
        for (int i = 0; i < n; i++) {
            // Get non-zero element range for current row
            const int start_idx = row_ptr[i];
            const int end_idx = row_ptr[i + 1];
            
            // Process all non-zero elements in current row
            for (int k = start_idx; k < end_idx; k++) {
                const double val = values[k];
                const int j = col_indices[k];
                
                // Calculate pooling coordinates
                int I, J;
                if (i < t) {
                    I = i / (q + 1);
                } else {
                    I = (i - t) / q + p;
                }
                
                if (j < t) {
                    J = j / (q + 1);
                } else {
                    J = (j - t) / q + p;
                }
                
                // Ensure within valid range
                if (I >= 0 && I < m && J >= 0 && J < m) {
                    // Update based on operation type
                    if (op == PoolingOp::SUM) {
                        V_local[I][J] += val;
                    } else { // MAX
                        if (val > V_local[I][J]) {
                            V_local[I][J] = val;
                        }
                    }
                    C_local[I][J]++;
                }
            }
        }
        
        // Merge thread results into global matrix
        #pragma omp critical
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    if (op == PoolingOp::SUM) {
                        V[i][j] += V_local[i][j];
                    } else { // MAX
                        if (V_local[i][j] > V[i][j]) {
                            V[i][j] = V_local[i][j];
                        }
                    }
                    C[i][j] += C_local[i][j];
                }
            }
        }
    }
}


void std_normalize(std::vector<std::vector<double>> &V_std)
{
    double mean = 0.0;
    for (auto vec: V_std)
    {
        for (auto val: vec)
        {
            mean += val;
        }
    }
    mean = mean/(V_std.size()*V_std.size());

    double std_ = 0.0;
    for (auto vec: V_std)
    {
        for (auto val: vec)
        {
            std_ += (val - mean)*(val - mean);
        }
    }
    std_ = std::sqrt(std_/(V_std.size()*V_std.size()));
    

    for (auto &vec: V_std)
    {
        for (auto &val: vec)
        {
            val = (val - mean)/std_;
        }
    }
}


void pool_dataset(std::ifstream &input_file, std::ofstream &out_file, int m, PoolingOp op)
{
    std::string line;
    int line_num = 0;

    while(std::getline(input_file, line))
    {
        line_num++;
        std::vector<std::vector<double>> V;
        std::vector<std::vector<int>> C;
        std::vector<double> row_data;
        std::stringstream ss(line);
        std::string cell;

        // Parse CSV line
        while(std::getline(ss, cell, ','))
        {
            try 
            {
                double value = std::stod(cell);
                row_data.push_back(value);
            } 
            catch (const std::invalid_argument& e) 
            {
                std::cerr << "Warning: Invalid value at line " << line_num 
                          << ", cell: '" << cell << "' - using 0.0\n";
                row_data.push_back(0.0);
            }            
        }

        // Check minimum data requirement
        if (row_data.size() < 6) {
            std::cerr << "Error: Line " << line_num << " has only " 
                      << row_data.size() << " elements (minimum 6 required). Skipping.\n";
            continue;
        }

        // Extract metadata
        int n_rows = static_cast<int>(row_data[0]);
        double theta = row_data[2];  // Corrected index
        double rho = row_data[3];    // Corrected index
        double h = row_data[4];      // Corrected index
        unsigned int nnz = static_cast<unsigned int>(row_data[5]);  // Number of non-zero elements
        unsigned int row_ptr_size = n_rows + 1;  // row_ptr array size
        
        // Verify data volume
        unsigned int min_required = 6 + nnz + nnz + row_ptr_size;
        if (row_data.size() < min_required) {
            std::cerr << "Error: Line " << line_num << " requires at least " 
                      << min_required << " elements but has only " 
                      << row_data.size() << ". Skipping.\n";
            continue;
        }
        
        // Extract CSR format data
        std::vector<double> values;
        std::vector<int> col_indices;
        std::vector<int> row_ptr;
        
        // Extract non-zero values (starting from index 6)
        for (unsigned int i = 6; i < 6 + nnz; ++i) {
            values.push_back(row_data[i]);
        }

        // Extract row pointers (starting from index 6 + nnz + row_ptr_size)
        for (unsigned int i = 6 + nnz; i < 6 + nnz + row_ptr_size; ++i) {
            row_ptr.push_back(static_cast<int>(row_data[i]));
        }        
        
        // Extract column indices (starting from index 6 + nnz + row_ptr_size)
        for (unsigned int i = 6 + nnz + row_ptr_size; i < 6 + 2*nnz + row_ptr_size; ++i) {
            col_indices.push_back(static_cast<int>(row_data[i]));
        }
        
        
        // Validate row pointers
        if (row_ptr.size() != static_cast<size_t>(n_rows + 1) || row_ptr.back() != static_cast<int>(nnz)) {
            std::cerr << "Error: Invalid row_ptr at line " << line_num 
                      << ". Expected size: " << (n_rows + 1)
                      << ", got: " << row_ptr.size()
                      << ". Last element: " << row_ptr.back()
                      << ", expected: " << nnz << ". Skipping.\n";
            continue;
        }
        
        // Validate index ranges
        bool invalid_index = false;
        for (int col_idx : col_indices) {
            if (col_idx < 0 || col_idx >= n_rows) {
                std::cerr << "Warning: Column index " << col_idx << " out of range [0, " 
                          << n_rows - 1 << "] at line " << line_num << "\n";
                invalid_index = true;
            }
        }
        
        if (invalid_index) {
            std::cerr << "Skipping line " << line_num << " due to invalid indices.\n";
            continue;
        }
        
        // Execute pooling (using CSR format)
        try {
            parallel_pooling_csr(values, col_indices, row_ptr, n_rows, m, op, V, C);
            std_normalize(V);
            
            // Write results
            out_file << rho << "," << theta << "," << h;
            for (const auto& vec : V) {
                for (double value : vec) {
                    out_file << "," << value;
                }
            }
            out_file << "\n";
        } 
        catch (const std::exception& e) {
            std::cerr << "Error during pooling at line " << line_num 
                      << ": " << e.what() << ". Skipping.\n";
        }
    }
}