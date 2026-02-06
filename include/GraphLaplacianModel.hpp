/* ---------------------------------------------------------------------
 * GraphLaplacianModel.hpp
 *
 * Graph-based matrix generation for AMG learning
 * Supports: Graph Laplacian, Spectral Clustering
 *
 * Implementation includes:
 * - Delaunay triangulation (Bowyer-Watson algorithm)
 * - Graph Laplacian generation (lognormal weights)
 * - Spectral Clustering (k-NN graphs)
 * - CSR format output compatible with AMGOperators
 * ---------------------------------------------------------------------
 */

#ifndef GRAPHLAPLACIANMODEL_HPP
#define GRAPHLAPLACIANMODEL_HPP

#include "AMGOperators.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <set>
#include <limits>
#include <stdexcept>

namespace GraphLaplacian {

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

    bool contains_vertex(int v) const {
        return v0 == v || v1 == v || v2 == v;
    }

    bool shares_edge_with(const Triangle& other) const {
        int shared = 0;
        if (contains_vertex(other.v0)) shared++;
        if (contains_vertex(other.v1)) shared++;
        if (contains_vertex(other.v2)) shared++;
        return shared == 2;
    }
};

// ============================================================================
// Delaunay Triangulation (Bowyer-Watson Algorithm)
// ============================================================================

class DelaunayTriangulation {
public:
    explicit DelaunayTriangulation(const std::vector<Point2D>& points);

    const std::vector<Triangle>& triangles() const { return triangles_; }
    const std::vector<Point2D>& points() const { return points_; }

    // Get adjacency information
    std::vector<std::vector<int>> get_neighbors() const;
    std::vector<std::pair<int, int>> get_edges() const;

private:
    void triangulate();  // Main Bowyer-Watson algorithm
    bool in_circumcircle(const Point2D& p, const Triangle& tri) const;
    double orient2d(const Point2D& a, const Point2D& b, const Point2D& c) const;
    Point2D circumcenter(const Triangle& tri) const;
    double circumradius_squared(const Triangle& tri) const;
    bool is_super_triangle_vertex(int v) const;

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
    if (points_.size() < 3) {
        throw std::runtime_error("Need at least 3 points for triangulation");
    }

    // Add super-triangle that contains all points
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
    double dmax = std::max(dx, dy);
    double midX = (minX + maxX) / 2.0;
    double midY = (minY + maxY) / 2.0;

    // Super-triangle vertices (far outside the point cloud)
    Point2D st1(midX - 20 * dmax, midY - dmax);
    Point2D st2(midX, midY + 20 * dmax);
    Point2D st3(midX + 20 * dmax, midY - dmax);

    int st1_idx = points_.size();
    int st2_idx = points_.size() + 1;
    int st3_idx = points_.size() + 2;

    points_.push_back(st1);
    points_.push_back(st2);
    points_.push_back(st3);

    // Initialize with super-triangle
    triangles_.push_back(Triangle(st1_idx, st2_idx, st3_idx));

    // Bowyer-Watson algorithm: insert points one by one
    for (int i = 0; i < num_original_points_; ++i) {
        const Point2D& p = points_[i];

        std::vector<Triangle> bad_triangles;
        std::vector<std::pair<int, int>> polygon;

        // Find bad triangles (whose circumcircle contains p)
        for (const auto& tri : triangles_) {
            if (in_circumcircle(p, tri)) {
                bad_triangles.push_back(tri);
            }
        }

        // Find the boundary of the bad triangles (polygon edges)
        for (const auto& tri : bad_triangles) {
            std::vector<std::pair<int, int>> edges = {
                {tri.v0, tri.v1},
                {tri.v1, tri.v2},
                {tri.v2, tri.v0}
            };

            for (const auto& edge : edges) {
                bool is_boundary = true;

                // Check if this edge is shared with another bad triangle
                for (const auto& other_tri : bad_triangles) {
                    if (&tri == &other_tri) continue;

                    if ((other_tri.contains_vertex(edge.first) &&
                         other_tri.contains_vertex(edge.second))) {
                        is_boundary = false;
                        break;
                    }
                }

                if (is_boundary) {
                    polygon.push_back(edge);
                }
            }
        }

        // Remove bad triangles
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

        // Add new triangles from point to polygon
        for (const auto& edge : polygon) {
            triangles_.push_back(Triangle(edge.first, edge.second, i));
        }
    }

    // Remove triangles that contain super-triangle vertices
    triangles_.erase(
        std::remove_if(triangles_.begin(), triangles_.end(),
            [this](const Triangle& t) {
                return is_super_triangle_vertex(t.v0) ||
                       is_super_triangle_vertex(t.v1) ||
                       is_super_triangle_vertex(t.v2);
            }),
        triangles_.end()
    );

    // Remove super-triangle vertices from points
    points_.resize(num_original_points_);
}

bool DelaunayTriangulation::in_circumcircle(const Point2D& p, const Triangle& tri) const {
    Point2D center = circumcenter(tri);
    double radius_sq = circumradius_squared(tri);
    return p.squared_distance_to(center) < radius_sq + 1e-10;  // Small epsilon for numerical stability
}

double DelaunayTriangulation::orient2d(const Point2D& a, const Point2D& b, const Point2D& c) const {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

Point2D DelaunayTriangulation::circumcenter(const Triangle& tri) const {
    const Point2D& a = points_[tri.v0];
    const Point2D& b = points_[tri.v1];
    const Point2D& c = points_[tri.v2];

    double D = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

    if (std::abs(D) < 1e-12) {
        // Degenerate triangle, return centroid
        return Point2D((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0);
    }

    double a_sq = a.x * a.x + a.y * a.y;
    double b_sq = b.x * b.x + b.y * b.y;
    double c_sq = c.x * c.x + c.y * c.y;

    double ux = (a_sq * (b.y - c.y) + b_sq * (c.y - a.y) + c_sq * (a.y - b.y)) / D;
    double uy = (a_sq * (c.x - b.x) + b_sq * (a.x - c.x) + c_sq * (b.x - a.x)) / D;

    return Point2D(ux, uy);
}

double DelaunayTriangulation::circumradius_squared(const Triangle& tri) const {
    Point2D center = circumcenter(tri);
    return points_[tri.v0].squared_distance_to(center);
}

bool DelaunayTriangulation::is_super_triangle_vertex(int v) const {
    return v >= num_original_points_;
}

std::vector<std::vector<int>> DelaunayTriangulation::get_neighbors() const {
    std::vector<std::vector<int>> neighbors(num_original_points_);

    for (const auto& tri : triangles_) {
        // Add bidirectional edges
        if (tri.v0 < num_original_points_ && tri.v1 < num_original_points_) {
            neighbors[tri.v0].push_back(tri.v1);
            neighbors[tri.v1].push_back(tri.v0);
        }
        if (tri.v1 < num_original_points_ && tri.v2 < num_original_points_) {
            neighbors[tri.v1].push_back(tri.v2);
            neighbors[tri.v2].push_back(tri.v1);
        }
        if (tri.v2 < num_original_points_ && tri.v0 < num_original_points_) {
            neighbors[tri.v2].push_back(tri.v0);
            neighbors[tri.v0].push_back(tri.v2);
        }
    }

    // Remove duplicates
    for (auto& neighbor_list : neighbors) {
        std::sort(neighbor_list.begin(), neighbor_list.end());
        neighbor_list.erase(std::unique(neighbor_list.begin(), neighbor_list.end()),
                          neighbor_list.end());
    }

    return neighbors;
}

std::vector<std::pair<int, int>> DelaunayTriangulation::get_edges() const {
    std::set<std::pair<int, int>> edge_set;

    for (const auto& tri : triangles_) {
        if (tri.v0 < num_original_points_ && tri.v1 < num_original_points_) {
            edge_set.insert({std::min(tri.v0, tri.v1), std::max(tri.v0, tri.v1)});
        }
        if (tri.v1 < num_original_points_ && tri.v2 < num_original_points_) {
            edge_set.insert({std::min(tri.v1, tri.v2), std::max(tri.v1, tri.v2)});
        }
        if (tri.v2 < num_original_points_ && tri.v0 < num_original_points_) {
            edge_set.insert({std::min(tri.v2, tri.v0), std::max(tri.v2, tri.v0)});
        }
    }

    return std::vector<std::pair<int, int>>(edge_set.begin(), edge_set.end());
}

// ============================================================================
// Graph Laplacian Configuration
// ============================================================================

enum class GraphType {
    LOGNORMAL_LAPLACIAN,           // Delaunay with lognormal weights
    UNIFORM_LAPLACIAN,             // Delaunay with uniform weights
    POISSON_GRID,                  // Regular grid, 5-point stencil
    ANISOTROPIC_DIFFUSION,         // Anisotropic grid (high aspect ratio)
    SPECTRAL_CLUSTERING            // k-NN graph with Gaussian similarity
};

struct GraphConfig {
    GraphType type = GraphType::LOGNORMAL_LAPLACIAN;
    int num_points = 64;           // Number of nodes
    double log_std = 1.0;          // For lognormal weights
    int k_neighbors = 10;          // For k-NN (spectral clustering)
    double sigma = 0.1;            // Gaussian kernel width
    double epsilon_x = 1.0;        // Anisotropy in x
    double epsilon_y = 0.01;       // Anisotropy in y
    int seed = 42;                 // Random seed
};

// ============================================================================
// Graph Laplacian Generator
// ============================================================================

class GraphLaplacianGenerator {
public:
    explicit GraphLaplacianGenerator(const GraphConfig& config);

    // Generate a single matrix
    AMGOperators::CSRMatrix generate();

    // Generate multiple matrices (with seed variations)
    std::vector<AMGOperators::CSRMatrix> generate_batch(int num_matrices);

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
    // Type-specific generators
    AMGOperators::CSRMatrix generate_lognormal_laplacian();
    AMGOperators::CSRMatrix generate_uniform_laplacian();
    AMGOperators::CSRMatrix generate_poisson_grid();
    AMGOperators::CSRMatrix generate_anisotropic_diffusion();
    AMGOperators::CSRMatrix generate_spectral_clustering();

    // Utility methods
    std::vector<Point2D> generate_random_points(int n);
    std::vector<double> generate_lognormal_weights(int n, std::mt19937& rng);
    std::vector<std::vector<int>> compute_knn(const std::vector<Point2D>& points, int k);
    double gaussian_similarity(const Point2D& p1, const Point2D& p2, double sigma);

    AMGOperators::CSRMatrix build_laplacian_from_edges(
        int num_nodes,
        const std::vector<std::pair<int, int>>& edges,
        const std::vector<double>& weights
    );

    GraphConfig config_;
    std::mt19937 rng_;
};

GraphLaplacianGenerator::GraphLaplacianGenerator(const GraphConfig& config)
    : config_(config), rng_(config.seed)
{}

AMGOperators::CSRMatrix GraphLaplacianGenerator::generate() {
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

std::vector<AMGOperators::CSRMatrix> GraphLaplacianGenerator::generate_batch(int num_matrices) {
    std::vector<AMGOperators::CSRMatrix> matrices;
    matrices.reserve(num_matrices);

    for (int i = 0; i < num_matrices; ++i) {
        set_seed(config_.seed + i);
        matrices.push_back(generate());
    }

    return matrices;
}

AMGOperators::CSRMatrix GraphLaplacianGenerator::generate_lognormal_laplacian() {
    // Generate random points in [0,1]²
    std::vector<Point2D> points = generate_random_points(config_.num_points);

    // Delaunay triangulation
    DelaunayTriangulation delaunay(points);
    std::vector<std::pair<int, int>> edges = delaunay.get_edges();

    // Generate lognormal weights for edges
    std::vector<double> weights = generate_lognormal_weights(edges.size(), rng_);

    // Build Laplacian matrix
    return build_laplacian_from_edges(config_.num_points, edges, weights);
}

AMGOperators::CSRMatrix GraphLaplacianGenerator::generate_uniform_laplacian() {
    // Generate random points
    std::vector<Point2D> points = generate_random_points(config_.num_points);

    // Delaunay triangulation
    DelaunayTriangulation delaunay(points);
    std::vector<std::pair<int, int>> edges = delaunay.get_edges();

    // Uniform weights
    std::vector<double> weights(edges.size(), 1.0);

    // Build Laplacian matrix
    return build_laplacian_from_edges(config_.num_points, edges, weights);
}

AMGOperators::CSRMatrix GraphLaplacianGenerator::generate_poisson_grid() {
    // Regular grid with 5-point stencil
    int n = static_cast<int>(std::sqrt(config_.num_points));
    int num_nodes = n * n;

    std::vector<std::pair<int, int>> edges;
    std::vector<double> weights;

    // Build grid connectivity
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;

            // Right neighbor
            if (j < n - 1) {
                edges.push_back({idx, idx + 1});
                weights.push_back(1.0);
            }

            // Bottom neighbor
            if (i < n - 1) {
                edges.push_back({idx, idx + n});
                weights.push_back(1.0);
            }
        }
    }

    return build_laplacian_from_edges(num_nodes, edges, weights);
}

AMGOperators::CSRMatrix GraphLaplacianGenerator::generate_anisotropic_diffusion() {
    // Anisotropic grid (different weights in x and y directions)
    int n = static_cast<int>(std::sqrt(config_.num_points));
    int num_nodes = n * n;

    std::vector<std::pair<int, int>> edges;
    std::vector<double> weights;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;

            // Right neighbor (x-direction)
            if (j < n - 1) {
                edges.push_back({idx, idx + 1});
                weights.push_back(config_.epsilon_x);
            }

            // Bottom neighbor (y-direction)
            if (i < n - 1) {
                edges.push_back({idx, idx + n});
                weights.push_back(config_.epsilon_y);
            }
        }
    }

    return build_laplacian_from_edges(num_nodes, edges, weights);
}

AMGOperators::CSRMatrix GraphLaplacianGenerator::generate_spectral_clustering() {
    // k-NN graph with Gaussian similarity
    std::vector<Point2D> points = generate_random_points(config_.num_points);
    std::vector<std::vector<int>> knn = compute_knn(points, config_.k_neighbors);

    std::vector<std::pair<int, int>> edges;
    std::vector<double> weights;

    for (int i = 0; i < config_.num_points; ++i) {
        for (int neighbor : knn[i]) {
            if (i < neighbor) {  // Avoid duplicates
                edges.push_back({i, neighbor});
                double sim = gaussian_similarity(points[i], points[neighbor], config_.sigma);
                weights.push_back(sim);
            }
        }
    }

    return build_laplacian_from_edges(config_.num_points, edges, weights);
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

std::vector<double> GraphLaplacianGenerator::generate_lognormal_weights(int n, std::mt19937& rng) {
    std::normal_distribution<double> normal_dist(0.0, config_.log_std);
    std::vector<double> weights;
    weights.reserve(n);

    for (int i = 0; i < n; ++i) {
        double log_weight = normal_dist(rng);
        weights.push_back(std::exp(log_weight));
    }

    return weights;
}

std::vector<std::vector<int>> GraphLaplacianGenerator::compute_knn(
    const std::vector<Point2D>& points, int k)
{
    int n = points.size();
    std::vector<std::vector<int>> knn(n);

    for (int i = 0; i < n; ++i) {
        // Compute distances to all other points
        std::vector<std::pair<double, int>> distances;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                double dist = points[i].squared_distance_to(points[j]);
                distances.push_back({dist, j});
            }
        }

        // Sort by distance and take k nearest
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

AMGOperators::CSRMatrix GraphLaplacianGenerator::build_laplacian_from_edges(
    int num_nodes,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<double>& weights)
{
    // Build adjacency list with weights
    std::vector<std::vector<std::pair<int, double>>> adj(num_nodes);

    for (size_t i = 0; i < edges.size(); ++i) {
        int u = edges[i].first;
        int v = edges[i].second;
        double w = weights[i];

        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // Build CSR Laplacian: L_ii = sum(weights), L_ij = -weight
    AMGOperators::CSRMatrix L;
    L.n_rows = num_nodes;
    L.n_cols = num_nodes;

    L.row_ptr.push_back(0);

    for (int i = 0; i < num_nodes; ++i) {
        // Sort neighbors for CSR format
        std::sort(adj[i].begin(), adj[i].end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        double degree = 0.0;
        for (const auto& [neighbor, weight] : adj[i]) {
            degree += weight;
        }

        // Add off-diagonal entries (negative)
        for (const auto& [neighbor, weight] : adj[i]) {
            if (neighbor < i) {
                L.col_indices.push_back(neighbor);
                L.values.push_back(-weight);
            }
        }

        // Add diagonal entry
        L.col_indices.push_back(i);
        L.values.push_back(degree);

        // Add remaining off-diagonal entries
        for (const auto& [neighbor, weight] : adj[i]) {
            if (neighbor > i) {
                L.col_indices.push_back(neighbor);
                L.values.push_back(-weight);
            }
        }

        L.row_ptr.push_back(L.values.size());
    }

    return L;
}

// ============================================================================
// Utility Functions
// ============================================================================

// Compute mesh size (average edge length)
double compute_mesh_size(const AMGOperators::CSRMatrix& A, int num_points) {
    // For graph matrices, use average of off-diagonal absolute values as proxy
    double sum_abs = 0.0;
    int count = 0;

    for (size_t i = 0; i < A.n_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] != static_cast<int>(i)) {  // Off-diagonal
                sum_abs += std::abs(A.values[j]);
                count++;
            }
        }
    }

    if (count == 0) return 1.0 / std::sqrt(num_points);

    // Average edge weight (inverted as proxy for mesh size)
    double avg_weight = sum_abs / count;
    return 1.0 / std::sqrt(std::max(avg_weight, 1e-10));
}

// Find optimal theta using grid search
double find_optimal_theta_for_graph(const AMGOperators::CSRMatrix& A, double& best_rho) {
    std::vector<double> theta_candidates = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6};

    double best_theta = 0.25;
    best_rho = 1.0;  // Default (no convergence)
    double target_ratio = 0.4;  // Target coarsening ratio

    for (double theta : theta_candidates) {
        // Compute C/F splitting
        std::vector<int> cf_splitting = AMGOperators::classical_cf_splitting(A, theta);

        // Count coarse points
        int num_coarse = 0;
        for (int type : cf_splitting) {
            if (type == 1) num_coarse++;  // Coarse point
        }

        double ratio = static_cast<double>(num_coarse) / A.n_rows;

        // Prefer theta that gives ratio close to target
        double error = std::abs(ratio - target_ratio);
        double current_best_error = std::abs(static_cast<double>(best_rho) - target_ratio);

        if (error < current_best_error) {
            best_theta = theta;
            best_rho = ratio;  // Store ratio as proxy for rho
        }
    }

    // Convert ratio to approximate convergence factor
    // Better coarsening (ratio closer to target) gives better rho
    best_rho = 0.1 + 0.8 * std::abs(best_rho - target_ratio);

    return best_theta;
}

} // namespace GraphLaplacian

#endif // GRAPHLAPLACIANMODEL_HPP
