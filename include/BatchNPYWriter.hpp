#ifndef BATCH_NPY_WRITER_HPP
#define BATCH_NPY_WRITER_HPP

#include <vector>
#include <string>
#include <iostream>
#include "NPYWriter.hpp"

/**
 * @brief Batch writer for GNN theta prediction data in NPY format
 *
 * Collects samples and writes them as structured NPY arrays:
 * - edge_index.npy: (num_samples, 2, max_edges)
 * - edge_attr.npy: (num_samples, max_edges)
 * - num_edges.npy: (num_samples,)
 * - y.npy: (num_samples,)
 * - metadata.npy: (num_samples, 4) containing [n, rho, h, epsilon]
 */
class GNNThetaBatchWriter
{
public:
    /**
     * @brief Constructor
     * @param max_edges Maximum number of edges per sample (for padding)
     * @param reserve_size Number of samples to reserve memory for
     */
    GNNThetaBatchWriter(size_t max_edges, size_t reserve_size = 10000)
        : max_edges_(max_edges)
    {
        edge_indices_.reserve(reserve_size * 2 * max_edges);
        edge_attrs_.reserve(reserve_size * max_edges);
        num_edges_.reserve(reserve_size);
        y_values_.reserve(reserve_size);
        metadata_.reserve(reserve_size * 4);
    }

    /**
     * @brief Add a sample to the batch
     * @param edge_index Edge indices (flattened: [src1, dst1, src2, dst2, ...])
     * @param edge_attr Edge attributes
     * @param y Target value (theta)
     * @param n Grid size
     * @param rho Rho value
     * @param h Mesh size
     * @param epsilon Epsilon value
     */
    void add_sample(const std::vector<int> &edge_index,
                    const std::vector<double> &edge_attr,
                    double y, int n, double rho, double h, double epsilon)
    {
        size_t num_edges = edge_attr.size();

        if (num_edges > max_edges_)
        {
            std::cerr << "Warning: Sample has " << num_edges << " edges, exceeding max "
                      << max_edges_ << ". Truncating." << std::endl;
            num_edges = max_edges_;
        }

        // Store num_edges for this sample
        num_edges_.push_back(static_cast<int>(num_edges));

        // Add edge_index with padding
        for (size_t i = 0; i < num_edges; ++i)
        {
            edge_indices_.push_back(edge_index[2 * i]);     // source
            edge_indices_.push_back(edge_index[2 * i + 1]); // target
        }
        // Pad with -1
        for (size_t i = num_edges; i < max_edges_; ++i)
        {
            edge_indices_.push_back(-1);
            edge_indices_.push_back(-1);
        }

        // Add edge_attr with padding
        for (size_t i = 0; i < num_edges; ++i)
        {
            edge_attrs_.push_back(edge_attr[i]);
        }
        // Pad with 0.0
        for (size_t i = num_edges; i < max_edges_; ++i)
        {
            edge_attrs_.push_back(0.0);
        }

        // Add target value
        y_values_.push_back(y);

        // Add metadata
        metadata_.push_back(static_cast<double>(n));
        metadata_.push_back(rho);
        metadata_.push_back(h);
        metadata_.push_back(epsilon);

        num_samples_++;
    }

    /**
     * @brief Write all collected samples to NPY files
     * @param output_dir Directory to write files to
     */
    void write(const std::string &output_dir)
    {
        if (num_samples_ == 0)
        {
            std::cerr << "Warning: No samples to write" << std::endl;
            return;
        }

        // Write edge_index as (num_samples, 2, max_edges)
        NPYWriter::write_array_2d(output_dir + "/edge_index.npy",
                                  reinterpret_cast<std::vector<double> &>(edge_indices_),
                                  num_samples_ * 2, max_edges_);

        // Write edge_attr as (num_samples, max_edges)
        NPYWriter::write_array_2d(output_dir + "/edge_attr.npy",
                                  edge_attrs_,
                                  num_samples_, max_edges_);

        // Write num_edges as (num_samples,)
        NPYWriter::write_array_1d(output_dir + "/num_edges.npy", num_edges_);

        // Write y as (num_samples,)
        NPYWriter::write_array_1d(output_dir + "/y.npy", y_values_);

        // Write metadata as (num_samples, 4)
        NPYWriter::write_array_2d(output_dir + "/metadata.npy",
                                  metadata_,
                                  num_samples_, 4);

        std::cout << "Wrote " << num_samples_ << " samples to " << output_dir << std::endl;
    }

    size_t get_num_samples() const { return num_samples_; }

private:
    size_t max_edges_;
    size_t num_samples_ = 0;

    std::vector<int> edge_indices_;      // Flattened: (num_samples * 2 * max_edges,)
    std::vector<double> edge_attrs_;     // Flattened: (num_samples * max_edges,)
    std::vector<int> num_edges_;         // (num_samples,)
    std::vector<double> y_values_;       // (num_samples,)
    std::vector<double> metadata_;       // Flattened: (num_samples * 4,)
};

/**
 * @brief Batch writer for P-value prediction data in NPY format
 *
 * Similar structure but with additional C/F, P, and S matrix data
 */
class PValueBatchWriter
{
public:
    PValueBatchWriter(size_t max_nodes, size_t max_nnz_A, size_t max_nnz_P, size_t max_nnz_S,
                      size_t reserve_size = 10000)
        : max_nodes_(max_nodes), max_nnz_A_(max_nnz_A),
          max_nnz_P_(max_nnz_P), max_nnz_S_(max_nnz_S)
    {
        // Reserve memory for efficiency
        // This is a simplified version - full implementation would track all arrays
    }

    // Methods to add samples and write...
    // (Implementation similar to GNNThetaBatchWriter)

private:
    size_t max_nodes_;
    size_t max_nnz_A_;
    size_t max_nnz_P_;
    size_t max_nnz_S_;
    size_t num_samples_ = 0;

    // Arrays for A, P, S matrices, C/F splitting, etc.
};

#endif // BATCH_NPY_WRITER_HPP
