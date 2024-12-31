#ifndef VECTORDB_BRUTE_FORCE_H
#define VECTORDB_BRUTE_FORCE_H

#include "distance.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Neighbor {
    std::string label;
    float distance;
};

class BruteForceIndex {
public:
    explicit BruteForceIndex(
        std::size_t dimensions,
        DistanceMetric metric = DistanceMetric::Euclidean
    );

    void add(std::string label, Vector vector);
    void remove(const std::string& label);
    std::vector<Neighbor> search(const Vector& query, std::size_t k) const;
    std::size_t size() const noexcept;

private:
    std::size_t dimensions_;
    DistanceMetric metric_;
    std::unordered_map<std::string, Vector> vectors_;
};

#endif
