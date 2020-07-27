#include "ceres/aggregate_sum.h"

#include <Eigen/Dense>
#include <iostream>

namespace ceres {
namespace internal {
void aggregate_sum(int num_vectors,
                   int dim,
                   double* data,
                   ContextImpl* context,
                   int num_threads) {
  const int min_block_size = 128;

  int iter = 0;

  while (num_vectors > 1) {
    int num_next = (num_vectors + 1) / 2;
    int block_size = std::max(min_block_size, dim / (num_threads / num_next));

    int num_blocks = (dim + block_size - 1) / block_size;
    int total_blocks = num_blocks * num_next;

    std::cout << __FUNCTION__ << " " << num_vectors << " " << dim << " "
              << block_size << " " << num_threads << " " << num_next << " "
              << total_blocks << std::endl;

    ParallelFor(context, 0, total_blocks, num_threads, [&](int id) {
      int vector_id = id / num_blocks;
      int second_vector_id = vector_id + num_next;
      if (second_vector_id >= num_vectors) return;
      int block_id = id % num_blocks;
      int len = std::min(block_size, dim - (block_id * block_size));
      int offset = block_id * block_size;

      Eigen::Map<Eigen::VectorXd> target(data + offset + dim * vector_id, len);
      Eigen::Map<const Eigen::VectorXd> add(
          data + offset + dim * second_vector_id, len);
      target += add;
    });

    num_vectors = num_next;
    ++iter;
  }
}
}  // namespace internal
}  // namespace ceres
