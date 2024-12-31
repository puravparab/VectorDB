#include "hnsw.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <stdexcept>

HNSWIndex::Node::Node(Vector value, int node_level)
    : point(std::move(value)), level(node_level), links(node_level + 1) {}

HNSWIndex::HNSWIndex(
    std::size_t dimensions,
    std::size_t m,
    std::size_t ef_construction,
    DistanceMetric metric,
    unsigned int seed
)
    : dimensions_(dimensions),
      m_(m),
      ef_construction_(ef_construction),
      metric_(metric),
      random_(seed) {
    if (dimensions == 0) {
        throw std::invalid_argument("dimensions must be positive");
    }
    if (m < 2 || ef_construction < m) {
        throw std::invalid_argument("invalid HNSW construction parameters");
    }
}

void HNSWIndex::add(std::string label, Vector vector) {
    if (vector.dimension() != dimensions_) {
        throw std::invalid_argument("vector has the wrong dimensions");
    }
    if (nodes_.find(label) != nodes_.end()) {
        throw std::invalid_argument("label already exists");
    }
    const int level = random_level();
    const std::string inserted_label = label;
    nodes_.emplace(label, Node(std::move(vector), level));
    if (max_level_ < 0) {
        entry_point_ = inserted_label;
        max_level_ = level;
        return;
    }

    std::string entry = entry_point_;
    const Vector& point = nodes_.at(inserted_label).point;
    for (int layer = max_level_; layer > level; --layer) {
        entry = greedy_closest(point, entry, layer);
    }
    for (int layer = std::min(level, max_level_); layer >= 0; --layer) {
        const auto candidates = search_layer(point, entry, ef_construction_, layer);
        const std::size_t count = std::min(candidates.size(), max_connections(layer));
        for (std::size_t index = 0; index < count; ++index) {
            const std::string& neighbor = candidates[index].second;
            nodes_.at(inserted_label).links[layer].insert(neighbor);
            nodes_.at(neighbor).links[layer].insert(inserted_label);
            prune(neighbor, layer);
        }
        if (!candidates.empty()) {
            entry = candidates.front().second;
        }
    }
    if (level > max_level_) {
        entry_point_ = inserted_label;
        max_level_ = level;
    }
}

std::vector<Neighbor> HNSWIndex::search(
    const Vector& query,
    std::size_t k,
    std::size_t ef
) const {
    if (query.dimension() != dimensions_) {
        throw std::invalid_argument("query has the wrong dimensions");
    }
    if (k == 0 || ef < k) {
        throw std::invalid_argument("ef must be greater than or equal to positive k");
    }
    if (nodes_.empty()) {
        return {};
    }
    std::string entry = entry_point_;
    for (int layer = max_level_; layer > 0; --layer) {
        entry = greedy_closest(query, entry, layer);
    }
    const auto candidates = search_layer(query, entry, ef, 0);
    std::vector<Neighbor> results;
    const std::size_t count = std::min(k, candidates.size());
    results.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        results.push_back({candidates[index].second, candidates[index].first});
    }
    return results;
}

std::size_t HNSWIndex::size() const noexcept {
    return nodes_.size();
}

int HNSWIndex::random_level() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const double sample = std::max(distribution(random_), 1e-12);
    return static_cast<int>(-std::log(sample) / std::log(static_cast<double>(m_)));
}

std::string HNSWIndex::greedy_closest(
    const Vector& query,
    const std::string& entry,
    int level
) const {
    std::string best = entry;
    float best_distance = vector_distance(query, nodes_.at(best).point, metric_);
    bool improved = true;
    while (improved) {
        improved = false;
        for (const auto& neighbor : nodes_.at(best).links[level]) {
            const float candidate = vector_distance(query, nodes_.at(neighbor).point, metric_);
            if (candidate < best_distance) {
                best = neighbor;
                best_distance = candidate;
                improved = true;
                break;
            }
        }
    }
    return best;
}

std::vector<std::pair<float, std::string>> HNSWIndex::search_layer(
    const Vector& query,
    const std::string& entry,
    std::size_t ef,
    int level
) const {
    using Candidate = std::pair<float, std::string>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
    std::priority_queue<Candidate> best;
    std::unordered_set<std::string> visited;
    const float entry_distance = vector_distance(query, nodes_.at(entry).point, metric_);
    candidates.push({entry_distance, entry});
    best.push({entry_distance, entry});
    visited.insert(entry);

    while (!candidates.empty()) {
        const Candidate current = candidates.top();
        candidates.pop();
        if (best.size() >= ef && current.first > best.top().first) {
            break;
        }
        for (const auto& neighbor : nodes_.at(current.second).links[level]) {
            if (!visited.insert(neighbor).second) {
                continue;
            }
            const float candidate_distance = vector_distance(
                query,
                nodes_.at(neighbor).point,
                metric_
            );
            if (best.size() < ef || candidate_distance < best.top().first) {
                candidates.push({candidate_distance, neighbor});
                best.push({candidate_distance, neighbor});
                if (best.size() > ef) {
                    best.pop();
                }
            }
        }
    }

    std::vector<Candidate> results;
    results.reserve(best.size());
    while (!best.empty()) {
        results.push_back(best.top());
        best.pop();
    }
    std::sort(results.begin(), results.end());
    return results;
}

void HNSWIndex::prune(const std::string& label, int level) {
    auto& links = nodes_.at(label).links[level];
    const std::size_t limit = max_connections(level);
    if (links.size() <= limit) {
        return;
    }
    std::vector<std::pair<float, std::string>> ranked;
    ranked.reserve(links.size());
    for (const auto& neighbor : links) {
        ranked.push_back({
            vector_distance(nodes_.at(label).point, nodes_.at(neighbor).point, metric_),
            neighbor,
        });
    }
    std::sort(ranked.begin(), ranked.end());
    for (std::size_t index = limit; index < ranked.size(); ++index) {
        links.erase(ranked[index].second);
        nodes_.at(ranked[index].second).links[level].erase(label);
    }
}

std::size_t HNSWIndex::max_connections(int level) const noexcept {
    return level == 0 ? m_ * 2 : m_;
}
