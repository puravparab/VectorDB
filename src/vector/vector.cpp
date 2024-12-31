#include "vector.h"

#include <cmath>
#include <stdexcept>
#include <utility>

Vector::Vector(std::size_t dimension) : values_(dimension, 0.0F) {
    if (dimension == 0) {
        throw std::invalid_argument("vector dimension must be positive");
    }
}

Vector::Vector(std::vector<float> values) : values_(std::move(values)) {
    if (values_.empty()) {
        throw std::invalid_argument("vector must not be empty");
    }
}

std::size_t Vector::dimension() const noexcept {
    return values_.size();
}

const std::vector<float>& Vector::values() const noexcept {
    return values_;
}

float Vector::dot(const Vector& other) const {
    require_same_dimension(other);
    float result = 0.0F;
    for (std::size_t index = 0; index < values_.size(); ++index) {
        result += values_[index] * other.values_[index];
    }
    return result;
}

float Vector::euclidean_distance(const Vector& other) const {
    require_same_dimension(other);
    float squared = 0.0F;
    for (std::size_t index = 0; index < values_.size(); ++index) {
        const float difference = values_[index] - other.values_[index];
        squared += difference * difference;
    }
    return std::sqrt(squared);
}

float Vector::cosine_similarity(const Vector& other) const {
    require_same_dimension(other);
    const float left_norm = std::sqrt(dot(*this));
    const float right_norm = std::sqrt(other.dot(other));
    if (left_norm == 0.0F || right_norm == 0.0F) {
        throw std::domain_error("cosine similarity is undefined for a zero vector");
    }
    return dot(other) / (left_norm * right_norm);
}

void Vector::require_same_dimension(const Vector& other) const {
    if (dimension() != other.dimension()) {
        throw std::invalid_argument("vector dimensions must match");
    }
}
