#ifndef CERES_INTERNAL_AGGREGATE_SUM_H_
#define CERES_INTERNAL_AGGREGATE_SUM_H_

#include "ceres/parallel_for.h"

#define USE_COL_STRUCTURE

namespace ceres {
namespace internal {

void aggregate_sum(int num_vectors,
                   int dim,
                   double* data,
                   ContextImpl* context,
                   int num_threads);

}
}  // namespace ceres

#endif
