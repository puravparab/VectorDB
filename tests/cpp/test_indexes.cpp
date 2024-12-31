#include "index/brute_force.h"
#include "index/hnsw.h"
#include "vector/vector.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

namespace {
bool close(float left, float right) {
    return std::fabs(left - right) < 1e-5F;
}

void test_vector_operations() {
    const Vector left(std::vector<float>{1.0F, 2.0F});
    const Vector right(std::vector<float>{4.0F, 6.0F});
    assert(close(left.dot(right), 16.0F));
    assert(close(left.euclidean_distance(right), 5.0F));
    assert(close(left.cosine_similarity(left), 1.0F));
}

void test_brute_force() {
    BruteForceIndex index(2);
    index.add("origin", Vector(std::vector<float>{0.0F, 0.0F}));
    index.add("right", Vector(std::vector<float>{5.0F, 0.0F}));
    const auto results = index.search(Vector(std::vector<float>{4.0F, 0.0F}), 2);
    assert(results.size() == 2);
    assert(results[0].label == "right");
    assert(close(results[0].distance, 1.0F));
}

void test_hnsw() {
    HNSWIndex index(2, 8, 60, DistanceMetric::Euclidean, 7);
    for (int value = 0; value < 100; ++value) {
        index.add(
            std::to_string(value),
            Vector(std::vector<float>{static_cast<float>(value), static_cast<float>(value % 7)})
        );
    }
    const auto results = index.search(Vector(std::vector<float>{42.0F, 0.0F}), 5, 50);
    assert(results.size() == 5);
    assert(results[0].label == "42");
    assert(close(results[0].distance, 0.0F));
}
}  // namespace

int main() {
    test_vector_operations();
    test_brute_force();
    test_hnsw();
    return 0;
}
