#ifndef VECTORDB_HNSW_H
#define VECTORDB_HNSW_H

#include "distance.h"

#include <cstddef>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class HNSWIndex {
public:
    explicit HNSWIndex(
        std::size_t dimensions,
        std::size_t m = 16,
        std::size_t ef_construction = 200,
        DistanceMetric metric = DistanceMetric::Euclidean,
        unsigned int seed = 0
    );

    void add(std::string label, Vector vector);
    std::vector<Neighbor> search(
        const Vector& query,
        std::size_t k,
        std::size_t ef = 50
    ) const;
    std::size_t size() const noexcept;

private:
    struct Node {
        Node(Vector value, int node_level);

        Vector point;
        int level;
        std::vector<std::unordered_set<std::string>> links;
    };

    int random_level();
    std::string greedy_closest(
        const Vector& query,
        const std::string& entry,
        int level
    ) const;
    std::vector<std::pair<float, std::string>> search_layer(
        const Vector& query,
        const std::string& entry,
        std::size_t ef,
        int level
    ) const;
    void prune(const std::string& label, int level);
    std::size_t max_connections(int level) const noexcept;

    std::size_t dimensions_;
    std::size_t m_;
    std::size_t ef_construction_;
    DistanceMetric metric_;
    std::mt19937 random_;
    std::unordered_map<std::string, Node> nodes_;
    std::string entry_point_;
    int max_level_ = -1;
};

#endif
