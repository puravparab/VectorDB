#include "brute_force.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

BruteForceIndex::BruteForceIndex(std::size_t dimensions, DistanceMetric metric)
    : dimensions_(dimensions), metric_(metric) {
    if (dimensions == 0) {
        throw std::invalid_argument("dimensions must be positive");
    }
}

void BruteForceIndex::add(std::string label, Vector vector) {
    if (vector.dimension() != dimensions_) {
        throw std::invalid_argument("vector has the wrong dimensions");
    }
    const auto inserted = vectors_.emplace(std::move(label), std::move(vector));
    if (!inserted.second) {
        throw std::invalid_argument("label already exists");
    }
}

void BruteForceIndex::remove(const std::string& label) {
    if (vectors_.erase(label) == 0) {
        throw std::out_of_range("label does not exist");
    }
}

std::vector<Neighbor> BruteForceIndex::search(const Vector& query, std::size_t k) const {
    if (query.dimension() != dimensions_) {
        throw std::invalid_argument("query has the wrong dimensions");
    }
    if (k == 0) {
        throw std::invalid_argument("k must be positive");
    }
    std::vector<Neighbor> results;
    results.reserve(vectors_.size());
    for (const auto& entry : vectors_) {
        results.push_back({entry.first, vector_distance(query, entry.second, metric_)});
    }
    std::sort(results.begin(), results.end(), [](const Neighbor& left, const Neighbor& right) {
        if (left.distance == right.distance) {
            return left.label < right.label;
        }
        return left.distance < right.distance;
    });
    if (results.size() > k) {
        results.resize(k);
    }
    return results;
}

std::size_t BruteForceIndex::size() const noexcept {
    return vectors_.size();
}
