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

// 并行池化函数
void parallel_pooling_coo(const std::vector<double>& val,
                      const std::vector<int>& row,
                      const std::vector<int>& col,
                      int n, int m, PoolingOp op,
                      std::vector<std::vector<double>>& V,
                      std::vector<std::vector<int>>& C) {
    
    // 检查输入一致性
    if (val.size() != row.size() || val.size() != col.size()) {
        std::invalid_argument("val, row, col must have same size");
    }
    
    const int nnz = val.size();
    if (nnz == 0) {
        V = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
        C = std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
        return;
    }
    
    // 计算块划分参数
    const int q = n / m;          // 基础块大小
    const int p = n % m;          // 额外块数
    const int t = (q + 1) * p;    // 分界点
    
    // 初始化输出矩阵
    V = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
    C = std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
    
    // 方法1：线程私有矩阵归并（适合m较小的情况）
    if (m <= 100) {
        // 获取线程数
        int num_threads = omp_get_max_threads();
        
        // 每个线程的私有矩阵
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
                
                // 计算池化坐标
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
                
                // 确保在有效范围内
                if (I >= 0 && I < m && J >= 0 && J < m) {
                    // 根据操作类型更新
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
        
        // 合并线程私有矩阵
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
    // 方法2：原子操作（适合m较大的情况）
    else {
        #pragma omp parallel for
        for (int k = 0; k < nnz; k++) {
            int i = row[k];
            int j = col[k];
            
            // 计算池化坐标
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
            
            // 确保在有效范围内
            if (I >= 0 && I < m && J >= 0 && J < m) {
                // 根据操作类型使用不同同步策略
                if (op == PoolingOp::SUM) {
                    #pragma omp atomic
                    V[I][J] += val[k];
                    #pragma omp atomic
                    C[I][J]++;
                } else { // MAX
                    // 使用临界区保护最大值更新
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

// 支持CSR格式的并行池化函数
void parallel_pooling_csr(const std::vector<double>& values,
                          const std::vector<int>& col_indices,
                          const std::vector<int>& row_ptr,
                          int n, int m, PoolingOp op,
                          std::vector<std::vector<double>>& V,
                          std::vector<std::vector<int>>& C) 
{
    // 检查输入有效性
    if (values.size() != col_indices.size()) {
        throw std::invalid_argument("values and col_indices must have same size");
    }
    
    const int nnz = values.size();
    if (nnz == 0) {
        V = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
        C = std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
        return;
    }
    
    // 检查row_ptr有效性
    if (row_ptr.empty() || row_ptr.back() != nnz) {
        throw std::invalid_argument("Invalid row_ptr array");
    }
    
    // 计算块划分参数
    const int q = n / m;          // 基础块大小
    const int p = n % m;          // 额外块数
    const int t = (q + 1) * p;    // 分界点
    
    // 初始化输出矩阵
    V = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
    C = std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
    
    #pragma omp parallel
    {
        // 每个线程的私有矩阵
        std::vector<std::vector<double>> V_local(m, std::vector<double>(m, 
            (op == PoolingOp::MAX) ? -std::numeric_limits<double>::max() : 0.0));
        std::vector<std::vector<int>> C_local(m, std::vector<int>(m, 0));
        
        // 并行处理每一行
        #pragma omp for
        for (int i = 0; i < n; i++) {
            // 获取当前行的非零元素范围
            const int start_idx = row_ptr[i];
            const int end_idx = row_ptr[i + 1];
            
            // 处理当前行的所有非零元素
            for (int k = start_idx; k < end_idx; k++) {
                const double val = values[k];
                const int j = col_indices[k];
                
                // 计算池化坐标
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
                
                // 确保在有效范围内
                if (I >= 0 && I < m && J >= 0 && J < m) {
                    // 根据操作类型更新
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
        
        // 合并线程结果到全局矩阵
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

        // 解析CSV行
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

        // 检查最小数据量要求
        if (row_data.size() < 6) {
            std::cerr << "Error: Line " << line_num << " has only " 
                      << row_data.size() << " elements (minimum 6 required). Skipping.\n";
            continue;
        }

        // 提取元数据
        int n_rows = static_cast<int>(row_data[0]);
        double theta = row_data[2];  // 修正索引
        double rho = row_data[3];    // 修正索引
        double h = row_data[4];      // 修正索引
        unsigned int nnz = static_cast<unsigned int>(row_data[5]);  // 非零元素数量
        unsigned int row_ptr_size = n_rows + 1;  // row_ptr数组大小
        
        // 验证数据量
        unsigned int min_required = 6 + nnz + nnz + row_ptr_size;
        if (row_data.size() < min_required) {
            std::cerr << "Error: Line " << line_num << " requires at least " 
                      << min_required << " elements but has only " 
                      << row_data.size() << ". Skipping.\n";
            continue;
        }
        
        // 提取CSR格式数据
        std::vector<double> values;
        std::vector<int> col_indices;
        std::vector<int> row_ptr;
        
        // 提取非零值 (从索引5开始)
        for (unsigned int i = 6; i < 6 + nnz; ++i) {
            values.push_back(row_data[i]);
        }

        // 提取行指针 (从索引5+2*nnz开始)
        for (unsigned int i = 6 + nnz; i < 6 + nnz + row_ptr_size; ++i) {
            row_ptr.push_back(static_cast<int>(row_data[i]));
        }        
        
        // 提取列索引 (从索引5+nnz开始)
        for (unsigned int i = 6 + nnz + row_ptr_size; i < 6 + 2*nnz + row_ptr_size; ++i) {
            col_indices.push_back(static_cast<int>(row_data[i]));
        }
        
        
        // 验证行指针
        if (row_ptr.size() != static_cast<size_t>(n_rows + 1) || row_ptr.back() != static_cast<int>(nnz)) {
            std::cerr << "Error: Invalid row_ptr at line " << line_num 
                      << ". Expected size: " << (n_rows + 1)
                      << ", got: " << row_ptr.size()
                      << ". Last element: " << row_ptr.back()
                      << ", expected: " << nnz << ". Skipping.\n";
            continue;
        }
        
        // 验证索引范围
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
        
        // 执行池化 (使用CSR格式)
        try {
            parallel_pooling_csr(values, col_indices, row_ptr, n_rows, m, op, V, C);
            std_normalize(V);
            
            // 写入结果
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