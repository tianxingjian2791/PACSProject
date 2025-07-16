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
void parallel_pooling(const std::vector<double>& val,
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

    while(std::getline(input_file, line))
    {
        std::vector<std::vector<double>> V;
        std::vector<std::vector<int>> C;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while(std::getline(ss, cell, ','))
        {
            try 
            {
                double value = std::stod(cell);
                row.push_back(value);
            } catch (const std::invalid_argument&) 
            {
                // Ignore invalid value
                row.push_back(0.0);
            }            
        }

        int n_rows = static_cast<int>(row[0]);
        double theta = row[2];
        double rho = row[3];
        double h = row[4];
        unsigned int vals_size = static_cast<unsigned int>(row[5]);
        
        // 
        out_file << rho << "," << theta << "," << h;
        
        unsigned int idx1 = vals_size + 5;
        std::vector<double> values;
        for (unsigned int i = 6; i<=idx1; ++i)
        {
            values.push_back(row[i]);
        }
        
        unsigned int idx2 = vals_size + idx1;
        std::vector<int> row_indices;
        for (unsigned int i = idx1 + 1; i<=idx2; ++i)
        {
            row_indices.push_back(static_cast<int>(row[i]));
        }

        unsigned int idx3 = vals_size + idx2;
        std::vector<int> col_indices;
        for (unsigned int i = idx2 + 1; i<=idx3;++i)
        {
            col_indices.push_back(static_cast<int>(row[i]));
        }

        parallel_pooling(values, row_indices, col_indices, n_rows, m, op, V, C);
        std_normalize(V);

        for (auto vec: V)
        {
            for (auto value: vec)
            {
                out_file << "," << value;
            }
        }
        out_file << "\n";
    }
    
}