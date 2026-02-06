/* ---------------------------------------------------------------------
 * GraphLaplacianModelEigen.hpp
 *
 * Eigen-based graph Laplacian generation for maximum performance
 * Uses Eigen::SparseMatrix and Triplet format for efficient construction
 * ---------------------------------------------------------------------
 */

#ifndef GRAPHLAPLACIANMODELEIGEN_HPP
#define GRAPHLAPLACIANMODELEIGEN_HPP

#include "AMGOperators.hpp"
#include <Eigen/Sparse>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <set>

namespace GraphLaplacian {

// Eigen types
using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;  // Row-major for efficient CSR conversion
using Triplet = Eigen::Triplet<double>;

// ============================================================================
// Point and Geometry Structures
// ============================================================================

struct Point2D {
    double x, y;

    Point2D() : x(0), y(0) {}
    Point2D(double x_, double y_) : x(x_), y(y_) {}

    Point2D operator+(const Point2D& other) const {
        return Point2D(x + other.x, y + other.y);
    }

    Point2D operator-(const Point2D& other) const {
        return Point2D(x - other.x, y - other.y);
    }

    Point2D operator*(double s) const {
        return Point2D(x * s, y * s);
    }

    double norm() const {
        return std::sqrt(x * x + y * y);
    }

    double distance_to(const Point2D& other) const {
        return (*this - other).norm();
    }

    double squared_distance_to(const Point2D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return dx * dx + dy * dy;
    }
};

struct Triangle {
    int v0, v1, v2;

    Triangle(int a, int b, int c) : v0(a), v1(b), v2(c) {}
};

// ============================================================================
// Delaunay Triangulation (Bowyer-Watson Algorithm)
// ============================================================================

class DelaunayTriangulation {
public:
    explicit DelaunayTriangulation(const std::vector<Point2D>& points);

    const std::vector<Triangle>& triangles() const { return triangles_; }
    const std::vector<Point2D>& points() const { return points_; }

    std::vector<std::vector<int>> getNeighbors() const;
    std::vector<std::pair<int, int>> getEdges() const;

private:
    void triangulate();
    bool inCircumcircle(const Point2D& p, const Triangle& tri) const;
    double orient2d(const Point2D& a, const Point2D& b, const Point2D& c) const;

    std::vector<Point2D> points_;
    std::vector<Triangle> triangles_;
    int num_original_points_;
};

DelaunayTriangulation::DelaunayTriangulation(const std::vector<Point2D>& points)
    : points_(points), num_original_points_(points.size())
{
    triangulate();
}

void DelaunayTriangulation::triangulate() {
    int n = points_.size();
    if (n < 3) return;

    // Find bounding box
    double minX = points_[0].x, maxX = points_[0].x;
    double minY = points_[0].y, maxY = points_[0].y;
    for (const auto& p : points_) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }

    double dx = maxX - minX;
    double dy = maxY - minY;
    double dmax = std::max(dx, dy) * 20.0;
    double midx = (minX + maxX) / 2.0;
    double midy = (minY + maxY) / 2.0;

    // Create super-triangle
    std::vector<Point2D> all_points = points_;
    all_points.push_back(Point2D(midx - dmax, midy - dmax));
    all_points.push_back(Point2D(midx, midy + dmax));
    all_points.push_back(Point2D(midx + dmax, midy - dmax));

    auto original_points = points_;
    points_ = all_points;

    triangles_.clear();
    triangles_.push_back(Triangle(n, n + 1, n + 2));

    // Bowyer-Watson algorithm
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<int, int>> polygon_edges;
        std::vector<Triangle> bad_triangles;

        for (const auto& tri : triangles_) {
            if (inCircumcircle(all_points[i], tri)) {
                bad_triangles.push_back(tri);
            }
        }

        for (const auto& tri : bad_triangles) {
            std::array<std::pair<int, int>, 3> edges = {{
                {tri.v0, tri.v1},
                {tri.v1, tri.v2},
                {tri.v2, tri.v0}
            }};

            for (const auto& edge : edges) {
                bool shared = false;
                for (const auto& other : bad_triangles) {
                    if (&tri == &other) continue;

                    std::array<std::pair<int, int>, 3> other_edges = {{
                        {other.v0, other.v1},
                        {other.v1, other.v2},
                        {other.v2, other.v0}
                    }};

                    for (const auto& other_edge : other_edges) {
                        if ((edge.first == other_edge.first && edge.second == other_edge.second) ||
                            (edge.first == other_edge.second && edge.second == other_edge.first)) {
                            shared = true;
                            break;
                        }
                    }
                    if (shared) break;
                }
                if (!shared) {
                    polygon_edges.push_back(edge);
                }
            }
        }

        triangles_.erase(
            std::remove_if(triangles_.begin(), triangles_.end(),
                [&bad_triangles](const Triangle& t) {
                    return std::find_if(bad_triangles.begin(), bad_triangles.end(),
                        [&t](const Triangle& bt) {
                            return t.v0 == bt.v0 && t.v1 == bt.v1 && t.v2 == bt.v2;
                        }) != bad_triangles.end();
                }),
            triangles_.end()
        );

        for (const auto& edge : polygon_edges) {
            triangles_.push_back(Triangle(edge.first, edge.second, i));
        }
    }

    points_ = original_points;

    triangles_.erase(
        std::remove_if(triangles_.begin(), triangles_.end(),
            [n](const Triangle& t) {
                return t.v0 >= n || t.v1 >= n || t.v2 >= n;
            }),
        triangles_.end()
    );
}

bool DelaunayTriangulation::inCircumcircle(const Point2D& p, const Triangle& tri) const {
    const Point2D& a = points_[tri.v0];
    const Point2D& b = points_[tri.v1];
    const Point2D& c = points_[tri.v2];

    double ax = a.x - p.x;
    double ay = a.y - p.y;
    double bx = b.x - p.x;
    double by = b.y - p.y;
    double cx = c.x - p.x;
    double cy = c.y - p.y;

    double ab = ax * ax + ay * ay;
    double bc = bx * bx + by * by;
    double ca = cx * cx + cy * cy;

    double det = ax * (by * ca - bc * cy) - ay * (bx * ca - bc * cx) +
                 ab * (bx * cy - by * cx);

    double orient = orient2d(a, b, c);
    return (orient > 0) ? (det > 0) : (det < 0);
}

double DelaunayTriangulation::orient2d(const Point2D& a, const Point2D& b, const Point2D& c) const {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

std::vector<std::vector<int>> DelaunayTriangulation::getNeighbors() const {
    std::vector<std::set<int>> neighbor_sets(num_original_points_);

    for (const auto& tri : triangles_) {
        if (tri.v0 < num_original_points_ && tri.v1 < num_original_points_) {
            neighbor_sets[tri.v0].insert(tri.v1);
            neighbor_sets[tri.v1].insert(tri.v0);
        }
        if (tri.v1 < num_original_points_ && tri.v2 < num_original_points_) {
            neighbor_sets[tri.v1].insert(tri.v2);
            neighbor_sets[tri.v2].insert(tri.v1);
        }
        if (tri.v2 < num_original_points_ && tri.v0 < num_original_points_) {
            neighbor_sets[tri.v2].insert(tri.v0);
            neighbor_sets[tri.v0].insert(tri.v2);
        }
    }

    std::vector<std::vector<int>> neighbors(num_original_points_);
    for (int i = 0; i < num_original_points_; ++i) {
        neighbors[i] = std::vector<int>(neighbor_sets[i].begin(), neighbor_sets[i].end());
    }

    return neighbors;
}

std::vector<std::pair<int, int>> DelaunayTriangulation::getEdges() const {
    std::set<std::pair<int, int>> edge_set;

    for (const auto& tri : triangles_) {
        auto addEdge = [&edge_set, this](int a, int b) {
            if (a >= num_original_points_ || b >= num_original_points_) return;
            if (a > b) std::swap(a, b);
            edge_set.insert({a, b});
        };
        addEdge(tri.v0, tri.v1);
        addEdge(tri.v1, tri.v2);
        addEdge(tri.v2, tri.v0);
    }

    return std::vector<std::pair<int, int>>(edge_set.begin(), edge_set.end());
}

// ============================================================================
// Graph Laplacian Configuration
// ============================================================================

enum class GraphType {
    LOGNORMAL_LAPLACIAN,
    UNIFORM_LAPLACIAN,
    POISSON_GRID,
    ANISOTROPIC_DIFFUSION,
    SPECTRAL_CLUSTERING
};

struct GraphConfig {
    GraphType type = GraphType::LOGNORMAL_LAPLACIAN;
    int num_points = 64;
    double log_std = 1.0;
    int k_neighbors = 10;
    double sigma = 0.1;
    double epsilon_x = 1.0;
    double epsilon_y = 0.01;
    int seed = 42;
};

// ============================================================================
// Graph Laplacian Generator (Eigen-based)
// ============================================================================

class GraphLaplacianGenerator {
public:
    explicit GraphLaplacianGenerator(const GraphConfig& config);

    SpMat generate();
    std::vector<SpMat> generate_batch(int num_matrices);

    void set_seed(int seed) {
        config_.seed = seed;
        rng_.seed(seed);
    }

    void set_config(const GraphConfig& config) {
        config_ = config;
        rng_.seed(config_.seed);
    }

    const GraphConfig& config() const { return config_; }

private:
    SpMat generate_lognormal_laplacian();
    SpMat generate_uniform_laplacian();
    SpMat generate_poisson_grid();
    SpMat generate_anisotropic_diffusion();
    SpMat generate_spectral_clustering();

    std::vector<Point2D> generate_random_points(int n);
    std::vector<std::vector<int>> compute_knn(const std::vector<Point2D>& points, int k);
    double gaussian_similarity(const Point2D& p1, const Point2D& p2, double sigma);

    SpMat buildLaplacianFromNeighbors(const std::vector<std::vector<int>>& neighbors,
                                       bool lognormal_weights);

    GraphConfig config_;
    std::mt19937 rng_;
};

GraphLaplacianGenerator::GraphLaplacianGenerator(const GraphConfig& config)
    : config_(config), rng_(config.seed)
{}

SpMat GraphLaplacianGenerator::generate() {
    switch (config_.type) {
        case GraphType::LOGNORMAL_LAPLACIAN:
            return generate_lognormal_laplacian();
        case GraphType::UNIFORM_LAPLACIAN:
            return generate_uniform_laplacian();
        case GraphType::POISSON_GRID:
            return generate_poisson_grid();
        case GraphType::ANISOTROPIC_DIFFUSION:
            return generate_anisotropic_diffusion();
        case GraphType::SPECTRAL_CLUSTERING:
            return generate_spectral_clustering();
        default:
            throw std::runtime_error("Unknown graph type");
    }
}

std::vector<SpMat> GraphLaplacianGenerator::generate_batch(int num_matrices) {
    std::vector<SpMat> matrices(num_matrices);

    for (int i = 0; i < num_matrices; ++i) {
        set_seed(config_.seed + i);
        matrices[i] = generate();
    }

    return matrices;
}

SpMat GraphLaplacianGenerator::generate_lognormal_laplacian() {
    auto points = generate_random_points(config_.num_points);
    DelaunayTriangulation delaunay(points);
    auto neighbors = delaunay.getNeighbors();
    return buildLaplacianFromNeighbors(neighbors, true);
}

SpMat GraphLaplacianGenerator::generate_uniform_laplacian() {
    auto points = generate_random_points(config_.num_points);
    DelaunayTriangulation delaunay(points);
    auto neighbors = delaunay.getNeighbors();
    return buildLaplacianFromNeighbors(neighbors, false);
}

SpMat GraphLaplacianGenerator::generate_poisson_grid() {
    int n = static_cast<int>(std::sqrt(config_.num_points));
    int num_nodes = n * n;

    std::vector<Triplet> triplets;
    triplets.reserve(5 * num_nodes);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            double diag = 0.0;

            if (j > 0) {
                triplets.push_back(Triplet(idx, idx - 1, -1.0));
                diag += 1.0;
            }
            if (j < n - 1) {
                triplets.push_back(Triplet(idx, idx + 1, -1.0));
                diag += 1.0;
            }
            if (i > 0) {
                triplets.push_back(Triplet(idx, idx - n, -1.0));
                diag += 1.0;
            }
            if (i < n - 1) {
                triplets.push_back(Triplet(idx, idx + n, -1.0));
                diag += 1.0;
            }

            triplets.push_back(Triplet(idx, idx, diag));
        }
    }

    SpMat L(num_nodes, num_nodes);
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

SpMat GraphLaplacianGenerator::generate_anisotropic_diffusion() {
    int n = static_cast<int>(std::sqrt(config_.num_points));
    int num_nodes = n * n;

    std::vector<Triplet> triplets;
    triplets.reserve(5 * num_nodes);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            double diag = 0.0;

            if (j > 0) {
                triplets.push_back(Triplet(idx, idx - 1, -config_.epsilon_x));
                diag += config_.epsilon_x;
            }
            if (j < n - 1) {
                triplets.push_back(Triplet(idx, idx + 1, -config_.epsilon_x));
                diag += config_.epsilon_x;
            }
            if (i > 0) {
                triplets.push_back(Triplet(idx, idx - n, -config_.epsilon_y));
                diag += config_.epsilon_y;
            }
            if (i < n - 1) {
                triplets.push_back(Triplet(idx, idx + n, -config_.epsilon_y));
                diag += config_.epsilon_y;
            }

            triplets.push_back(Triplet(idx, idx, diag));
        }
    }

    SpMat L(num_nodes, num_nodes);
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

SpMat GraphLaplacianGenerator::generate_spectral_clustering() {
    auto points = generate_random_points(config_.num_points);
    auto knn = compute_knn(points, config_.k_neighbors);

    std::vector<Triplet> triplets;
    std::vector<double> diag(config_.num_points, 0.0);

    for (int i = 0; i < config_.num_points; ++i) {
        for (int j : knn[i]) {
            if (i < j) {
                double sim = gaussian_similarity(points[i], points[j], config_.sigma);
                triplets.push_back(Triplet(i, j, -sim));
                triplets.push_back(Triplet(j, i, -sim));
                diag[i] += sim;
                diag[j] += sim;
            }
        }
    }

    for (int i = 0; i < config_.num_points; ++i) {
        triplets.push_back(Triplet(i, i, diag[i]));
    }

    SpMat L(config_.num_points, config_.num_points);
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

std::vector<Point2D> GraphLaplacianGenerator::generate_random_points(int n) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<Point2D> points;
    points.reserve(n);

    for (int i = 0; i < n; ++i) {
        points.emplace_back(dist(rng_), dist(rng_));
    }

    return points;
}

std::vector<std::vector<int>> GraphLaplacianGenerator::compute_knn(
    const std::vector<Point2D>& points, int k)
{
    int n = points.size();
    std::vector<std::vector<int>> knn(n);

    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<double, int>> distances;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                double dist = points[i].squared_distance_to(points[j]);
                distances.push_back({dist, j});
            }
        }

        std::sort(distances.begin(), distances.end());
        int actual_k = std::min(k, static_cast<int>(distances.size()));

        for (int j = 0; j < actual_k; ++j) {
            knn[i].push_back(distances[j].second);
        }
    }

    return knn;
}

double GraphLaplacianGenerator::gaussian_similarity(
    const Point2D& p1, const Point2D& p2, double sigma)
{
    double dist_sq = p1.squared_distance_to(p2);
    return std::exp(-dist_sq / (2.0 * sigma * sigma));
}

SpMat GraphLaplacianGenerator::buildLaplacianFromNeighbors(
    const std::vector<std::vector<int>>& neighbors,
    bool lognormal_weights)
{
    int n = neighbors.size();
    std::normal_distribution<double> normal_dist(0.0, config_.log_std);
    std::uniform_real_distribution<double> uniform_dist(0.5, 1.5);

    std::vector<Triplet> triplets;
    std::vector<double> diag(n, 0.0);

    // Estimate size for efficiency
    int estimated_edges = 0;
    for (const auto& nb : neighbors) {
        estimated_edges += nb.size();
    }
    triplets.reserve(estimated_edges + n);

    for (int i = 0; i < n; ++i) {
        for (int j : neighbors[i]) {
            if (i < j) {
                double weight;
                if (lognormal_weights) {
                    weight = std::exp(normal_dist(rng_));
                } else {
                    weight = uniform_dist(rng_);
                }

                triplets.push_back(Triplet(i, j, -weight));
                triplets.push_back(Triplet(j, i, -weight));
                diag[i] += weight;
                diag[j] += weight;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        triplets.push_back(Triplet(i, i, diag[i]));
    }

    SpMat L(n, n);
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

// ============================================================================
// Conversion Functions: Eigen <-> AMGOperators::CSRMatrix
// ============================================================================

inline AMGOperators::CSRMatrix eigenToCSR(const SpMat& eigen_mat) {
    AMGOperators::CSRMatrix csr;
    csr.n_rows = eigen_mat.rows();
    csr.n_cols = eigen_mat.cols();

    csr.row_ptr.resize(csr.n_rows + 1);
    csr.col_indices.reserve(eigen_mat.nonZeros());
    csr.values.reserve(eigen_mat.nonZeros());

    csr.row_ptr[0] = 0;
    for (int i = 0; i < eigen_mat.outerSize(); ++i) {
        for (SpMat::InnerIterator it(eigen_mat, i); it; ++it) {
            csr.col_indices.push_back(it.col());
            csr.values.push_back(it.value());
        }
        csr.row_ptr[i + 1] = csr.values.size();
    }

    return csr;
}

// ============================================================================
// Utility Functions
// ============================================================================

inline double compute_mesh_size(const SpMat& A, int num_points) {
    double sum_abs = 0.0;
    int count = 0;

    for (int i = 0; i < A.outerSize(); ++i) {
        for (SpMat::InnerIterator it(A, i); it; ++it) {
            if (it.row() != it.col()) {
                sum_abs += std::abs(it.value());
                count++;
            }
        }
    }

    if (count == 0) return 1.0 / std::sqrt(num_points);

    double avg_weight = sum_abs / count;
    return 1.0 / std::sqrt(std::max(avg_weight, 1e-10));
}

inline double find_optimal_theta_for_graph(const SpMat& eigen_mat, double& best_rho) {
    // Fast heuristic-based theta selection (matching unified-amg-learning performance)
    // Uses sparsity-based estimation instead of expensive C/F splitting

    std::vector<double> theta_candidates = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6};

    double best_theta = 0.25;
    best_rho = 1.0;

    int n = eigen_mat.rows();
    int nnz = eigen_mat.nonZeros();
    int nnz_offdiag = nnz - n;  // Exclude diagonal

    for (double theta : theta_candidates) {
        // Count strong connections for this theta
        std::vector<double> max_offdiag(n, 0.0);

        // Find max off-diagonal magnitude per row
        for (int k = 0; k < eigen_mat.outerSize(); ++k) {
            for (typename SpMat::InnerIterator it(eigen_mat, k); it; ++it) {
                if (it.row() != it.col()) {
                    max_offdiag[it.row()] = std::max(max_offdiag[it.row()], std::abs(it.value()));
                }
            }
        }

        // Count strong connections
        int num_strong = 0;
        for (int k = 0; k < eigen_mat.outerSize(); ++k) {
            for (typename SpMat::InnerIterator it(eigen_mat, k); it; ++it) {
                if (it.row() != it.col()) {
                    double threshold = theta * max_offdiag[it.row()];
                    if (std::abs(it.value()) >= threshold) {
                        num_strong++;
                    }
                }
            }
        }

        // Estimate convergence factor from sparsity ratio
        double sparsity_ratio = (nnz_offdiag > 0) ? static_cast<double>(num_strong) / nnz_offdiag : 0.0;
        double rho = 0.3 + 0.5 * (1.0 - sparsity_ratio);

        // Select theta with lowest estimated rho
        if (rho < best_rho) {
            best_rho = rho;
            best_theta = theta;
        }
    }

    return best_theta;
}

} // namespace GraphLaplacian

#endif // GRAPHLAPLACIANMODELEIGEN_HPP
