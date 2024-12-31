#ifndef VECTORDB_VECTOR_H
#define VECTORDB_VECTOR_H

#include <cstddef>
#include <vector>

class Vector {
public:
    explicit Vector(std::size_t dimension);
    explicit Vector(std::vector<float> values);

    std::size_t dimension() const noexcept;
    const std::vector<float>& values() const noexcept;
    float dot(const Vector& other) const;
    float euclidean_distance(const Vector& other) const;
    float cosine_similarity(const Vector& other) const;

private:
    void require_same_dimension(const Vector& other) const;

    std::vector<float> values_;
};

#endif
