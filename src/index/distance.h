#ifndef VECTORDB_DISTANCE_H
#define VECTORDB_DISTANCE_H

#include "../vector/vector.h"

enum class DistanceMetric { Euclidean, Cosine };

inline float vector_distance(
    const Vector& left,
    const Vector& right,
    DistanceMetric metric
) {
    if (metric == DistanceMetric::Euclidean) {
        return left.euclidean_distance(right);
    }
    return 1.0F - left.cosine_similarity(right);
}

#endif
