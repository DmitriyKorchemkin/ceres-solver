// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "aggregate_sum.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/internal/eigen.h"
#include "ceres/parallel_for.h"
#include "ceres/partitioned_matrix_view.h"
#include "ceres/small_blas.h"
#include "glog/logging.h"

#define BENCH_TIMER(callee)                                                 \
  {                                                                         \
    AutoTimer timer;                                                        \
    callee;                                                                 \
    auto res = timer.lap();                                                 \
    std::cout << "\nISB: " << __FUNCTION__ << ", " << #callee << ", "       \
              << matrix_.num_nonzeros() << ", " << num_rows() << ", "       \
              << num_cols() << ", " << num_cols_e() << ", " << num_cols_f() \
              << ", " << res.first << ", " << res.second << std::endl;      \
  }

namespace ceres {
namespace internal {


template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    PartitionedMatrixView(const BlockSparseMatrix& matrix, int num_col_blocks_e)
    : matrix_(matrix), num_col_blocks_e_(num_col_blocks_e) {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  CHECK(bs != nullptr);

  num_col_blocks_f_ = bs->cols.size() - num_col_blocks_e_;

  // Compute the number of row blocks in E. The number of row blocks
  // in E maybe less than the number of row blocks in the input matrix
  // as some of the row blocks at the bottom may not have any
  // e_blocks. For a definition of what an e_block is, please see
  // explicit_schur_complement_solver.h
  num_row_blocks_e_ = 0;
  for (int r = 0; r < bs->rows.size(); ++r) {
    const std::vector<Cell>& cells = bs->rows[r].cells;
    if (cells[0].block_id < num_col_blocks_e_) {
      ++num_row_blocks_e_;
    }
  }

  // Compute the number of columns in E and F.
  num_cols_e_ = 0;
  num_cols_f_ = 0;

  for (int c = 0; c < bs->cols.size(); ++c) {
    const Block& block = bs->cols[c];
    if (c < num_col_blocks_e_) {
      num_cols_e_ += block.size;
    } else {
      num_cols_f_ += block.size;
    }
  }

  CHECK_EQ(num_cols_e_ + num_cols_f_, matrix_.num_cols());
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    ~PartitionedMatrixView() {}

// The next four methods don't seem to be particularly cache
// friendly. This is an artifact of how the BlockStructure of the
// input matrix is constructed. These methods will benefit from
// multithreading as well as improved data layout.

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    RightMultiplyE_sequential_row(const double* x, double* y) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();

    const double* values = matrix_.values();
    for (int r = 0; r < num_row_blocks_e_; ++r) {
      const Cell& cell = bs->rows[r].cells[0];
      const int row_block_pos = bs->rows[r].block.position;
      const int row_block_size = bs->rows[r].block.size;
      const int col_block_id = cell.block_id;
      const int col_block_pos = bs->cols[col_block_id].position;
      const int col_block_size = bs->cols[col_block_id].size;
      MatrixVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
          values + cell.position,
          row_block_size,
          col_block_size,
          x + col_block_pos,
          y + row_block_pos);
    }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    RightMultiplyE_parallel_row(const double* x, double* y) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  const double* values = matrix_.values();

  ParallelFor(matrix_.GetContext(),
              0,
              num_row_blocks_e_,
              matrix_.GetNumThreads(),
              [&](int r) {
                const Cell& cell = bs->rows[r].cells[0];
                const int row_block_pos = bs->rows[r].block.position;
                const int row_block_size = bs->rows[r].block.size;
                const int col_block_id = cell.block_id;
                const int col_block_pos = bs->cols[col_block_id].position;
                const int col_block_size = bs->cols[col_block_id].size;
                MatrixVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
                    values + cell.position,
                    row_block_size,
                    col_block_size,
                    x + col_block_pos,
                    y + row_block_pos);
              });
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    RightMultiplyE(const double* x, double* y) const {
  // Iterate over the first num_row_blocks_e_ row blocks, and multiply
  // by the first cell in each row block.
  auto context = matrix_.GetContext();
  if (!context) {
    RightMultiplyE_sequential_row(x, y);
  } else {
    const double* values = matrix_.values();
    std::unique_ptr<double[]> foo(new double[num_rows()]);
    memcpy(foo.get(), y, sizeof(double) * num_rows());

    BENCH_TIMER(RightMultiplyE_parallel_row(x, foo.get()));
    BENCH_TIMER(RightMultiplyE_parallel_row(x, y));
  }
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    RightMultiplyF_sequential_row(const double* x, double* y) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();

  // Iterate over row blocks, and if the row block is in E, then
  //   // multiply by all the cells except the first one which is of type
  //     // E. If the row block is not in E (i.e its in the bottom
  //       // num_row_blocks - num_row_blocks_e row blocks), then all the
  //       cells
  //         // are of type F and multiply by them all.
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const std::vector<Cell>& cells = bs->rows[r].cells;
    for (int c = 1; c < cells.size(); ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_pos = bs->cols[col_block_id].position;
      const int col_block_size = bs->cols[col_block_id].size;
      MatrixVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
          values + cells[c].position,
          row_block_size,
          col_block_size,
          x + col_block_pos - num_cols_e_,
          y + row_block_pos);
    }
  }

  for (int r = num_row_blocks_e_; r < bs->rows.size(); ++r) {
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const std::vector<Cell>& cells = bs->rows[r].cells;
    for (int c = 0; c < cells.size(); ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_pos = bs->cols[col_block_id].position;
      const int col_block_size = bs->cols[col_block_id].size;
      MatrixVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
          values + cells[c].position,
          row_block_size,
          col_block_size,
          x + col_block_pos - num_cols_e_,
          y + row_block_pos);
    }
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    RightMultiplyF_parallel_row(const double* x, double* y) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  const double* values = matrix_.values();
  ParallelFor(matrix_.GetContext(),
              0,
              num_row_blocks_e_,
              matrix_.GetNumThreads(),
              [&](int r) {
                const int row_block_pos = bs->rows[r].block.position;
                const int row_block_size = bs->rows[r].block.size;
                const std::vector<Cell>& cells = bs->rows[r].cells;
                for (int c = 1; c < cells.size(); ++c) {
                  const int col_block_id = cells[c].block_id;
                  const int col_block_pos = bs->cols[col_block_id].position;
                  const int col_block_size = bs->cols[col_block_id].size;
                  MatrixVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
                      values + cells[c].position,
                      row_block_size,
                      col_block_size,
                      x + col_block_pos - num_cols_e_,
                      y + row_block_pos);
                }
              });

  ParallelFor(matrix_.GetContext(),
              num_row_blocks_e_,
              bs->rows.size(),
              matrix_.GetNumThreads(),
              [&](int r) {
                const int row_block_pos = bs->rows[r].block.position;
                const int row_block_size = bs->rows[r].block.size;
                const std::vector<Cell>& cells = bs->rows[r].cells;
                for (int c = 0; c < cells.size(); ++c) {
                  const int col_block_id = cells[c].block_id;
                  const int col_block_pos = bs->cols[col_block_id].position;
                  const int col_block_size = bs->cols[col_block_id].size;
                  MatrixVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
                      values + cells[c].position,
                      row_block_size,
                      col_block_size,
                      x + col_block_pos - num_cols_e_,
                      y + row_block_pos);
                }
              });
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    RightMultiplyF(const double* x, double* y) const {
  // Iterate over row blocks, and if the row block is in E, then
  // multiply by all the cells except the first one which is of type
  // E. If the row block is not in E (i.e its in the bottom
  // num_row_blocks - num_row_blocks_e row blocks), then all the cells
  // are of type F and multiply by them all.

  auto context = matrix_.GetContext();
  if (!context) {
    RightMultiplyF_sequential_row(x, y);
  } else {
    const double* values = matrix_.values();
    std::unique_ptr<double[]> foo(new double[num_rows()]);
    memcpy(foo.get(), y, sizeof(double)*num_rows());

    BENCH_TIMER(RightMultiplyF_sequential_row(x, foo.get()));
    BENCH_TIMER(RightMultiplyF_parallel_row(x, y));
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyE_sequential_row(const double* x, double* y) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();

  // Iterate over the first num_row_blocks_e_ row blocks, and multiply
  // by the first cell in each row block.
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const Cell& cell = bs->rows[r].cells[0];
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const int col_block_id = cell.block_id;
    const int col_block_pos = bs->cols[col_block_id].position;
    const int col_block_size = bs->cols[col_block_id].size;

    // visited_cols.insert(col_block_pos);
    MatrixTransposeVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
        values + cell.position,
        row_block_size,
        col_block_size,
        x + row_block_pos,
        y + col_block_pos);
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyE_sequential_col(const double* x, double* y) const {
  const double* values = matrix_.values();
  const CompressedColumnBlockStructure* cs = matrix_.block_structure_columns();
  for (int c = 0; c < num_col_blocks_e_; ++c) {
    const int col_block_pos = cs->cols[c].block.position;
    const int col_block_size = cs->cols[c].block.size;
    for (auto& cell : cs->cols[c].cells) {
      const int row_block_id = cell.block_id;
      const int row_block_size = cs->rows[row_block_id].size;
      const int row_block_pos = cs->rows[row_block_id].position;

      MatrixTransposeVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
          values + cell.position,
          row_block_size,
          col_block_size,
          x + row_block_pos,
          y + col_block_pos);
    }
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyE_parallel_col(const double* x, double* y) const {
  const CompressedColumnBlockStructure* cs = matrix_.block_structure_columns();
  const double* values = matrix_.values();
  ParallelFor(matrix_.GetContext(),
              0,
              num_col_blocks_e_,
              matrix_.GetNumThreads(),
              [&](int c) {
                const int col_block_pos = cs->cols[c].block.position;
                const int col_block_size = cs->cols[c].block.size;
                for (auto& cell : cs->cols[c].cells) {
                  const int row_block_id = cell.block_id;
                  const int row_block_size = cs->rows[row_block_id].size;
                  const int row_block_pos = cs->rows[row_block_id].position;

                  MatrixTransposeVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
                      values + cell.position,
                      row_block_size,
                      col_block_size,
                      x + row_block_pos,
                      y + col_block_pos);
                }
              });
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyE_sequential_col_transpose(const double* x, double* y) const {
  const double* values = matrix_.values_transpose();
  const CompressedColumnBlockStructure* cs =
      matrix_.block_structure_columns_transpose();
  for (int c = 0; c < num_col_blocks_e_; ++c) {
    const int col_block_pos = cs->cols[c].block.position;
    const int col_block_size = cs->cols[c].block.size;
    for (auto& cell : cs->cols[c].cells) {
      const int row_block_id = cell.block_id;
      const int row_block_size = cs->rows[row_block_id].size;
      const int row_block_pos = cs->rows[row_block_id].position;

      MatrixTransposeVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
          values + cell.position,
          row_block_size,
          col_block_size,
          x + row_block_pos,
          y + col_block_pos);
    }
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyE_parallel_col_transpose(const double* x, double* y) const {
  const CompressedColumnBlockStructure* cs =
      matrix_.block_structure_columns_transpose();
  const double* values = matrix_.values_transpose();
  ParallelFor(matrix_.GetContext(),
              0,
              num_col_blocks_e_,
              matrix_.GetNumThreads(),
              [&](int c) {
                const int col_block_pos = cs->cols[c].block.position;
                const int col_block_size = cs->cols[c].block.size;
                for (auto& cell : cs->cols[c].cells) {
                  const int row_block_id = cell.block_id;
                  const int row_block_size = cs->rows[row_block_id].size;
                  const int row_block_pos = cs->rows[row_block_id].position;

                  MatrixTransposeVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
                      values + cell.position,
                      row_block_size,
                      col_block_size,
                      x + row_block_pos,
                      y + col_block_pos);
                }
              });
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyE_parallel_row(const double* x, double* y_) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();

  // Iterate over the first num_row_blocks_e_ row blocks, and multiply
  // by the first cell in each row block.

  int threads = matrix_.GetNumThreads();
  size_t total = num_cols_e() * threads;
  std::unique_ptr<double[]> acc_(new double[total * threads]);
  double* acc = acc_.get();
  memset(acc, 0x0, sizeof(double) * total);
  memcpy(acc, y_, sizeof(double) * num_cols_f());

  AutoTimer timer;
  const double* values = matrix_.values();
  ParallelFor(matrix_.GetContext(),
              0,
              num_row_blocks_e_,
              threads,
              [&](int thread_id, int r) {
                double* y = acc + thread_id * num_cols_e();

                const Cell& cell = bs->rows[r].cells[0];
                const int row_block_pos = bs->rows[r].block.position;
                const int row_block_size = bs->rows[r].block.size;
                const int col_block_id = cell.block_id;
                const int col_block_pos = bs->cols[col_block_id].position;
                const int col_block_size = bs->cols[col_block_id].size;

                // visited_cols.insert(col_block_pos);
                MatrixTransposeVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
                    values + cell.position,
                    row_block_size,
                    col_block_size,
                    x + row_block_pos,
                    y + col_block_pos);
              });

  auto compute = timer.lap();

  aggregate_sum(threads, num_cols_e(), acc, matrix_.GetContext(), threads);
  memcpy(y_, acc, sizeof(double) * num_cols_e());

  auto aggregate = timer.lap();

  std::cout << "\nISB-A: " << __FUNCTION__ << ", " << matrix_.num_nonzeros()
            << ", " << num_rows() << ", " << num_cols() << ", " << num_cols_e()
            << ", " << num_cols_f() << ", " << compute.first << ", "
            << compute.second << std::endl;
  std::cout << "\nISB-A: " << __FUNCTION__ << ", " << matrix_.num_nonzeros()
            << ", " << num_rows() << ", " << num_cols() << ", " << num_cols_e()
            << ", " << num_cols_f() << ", " << aggregate.first << ", "
            << aggregate.second << std::endl;
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyE(const double* x, double* y) const {
  auto context = matrix_.GetContext();
  if (!context) {
    LeftMultiplyE_sequential_row(x, y);
  } else {
    std::unique_ptr<double[]> foo(new double[num_cols_e()]);
    memcpy(foo.get(), y, sizeof(double) * num_cols_e());

    void(matrix_.CreateTransposedStructure());

    BENCH_TIMER(LeftMultiplyE_sequential_row(x, foo.get()));
    BENCH_TIMER(LeftMultiplyE_sequential_col(x, foo.get()));
    BENCH_TIMER(LeftMultiplyE_parallel_row(x, foo.get()));
    BENCH_TIMER(LeftMultiplyE_sequential_col_transpose(x, foo.get()));
    BENCH_TIMER(LeftMultiplyE_parallel_col_transpose(x, foo.get()));
    BENCH_TIMER(LeftMultiplyE_parallel_col(x, y));
  }
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyF_sequential_row(const double* x, double* y) const {
  // Iterate over row blocks, and if the row block is in E, then
  // multiply by all the cells except the first one which is of type
  // E. If the row block is not in E (i.e its in the bottom
  // num_row_blocks - num_row_blocks_e row blocks), then all the cells
  // are of type F and multiply by them all.
  const double* values = matrix_.values();
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const std::vector<Cell>& cells = bs->rows[r].cells;
    for (int c = 1; c < cells.size(); ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_pos = bs->cols[col_block_id].position;
      const int col_block_size = bs->cols[col_block_id].size;
      MatrixTransposeVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
          values + cells[c].position,
          row_block_size,
          col_block_size,
          x + row_block_pos,
          y + col_block_pos - num_cols_e_);
    }
    }

    for (int r = num_row_blocks_e_; r < bs->rows.size(); ++r) {
      const int row_block_pos = bs->rows[r].block.position;
      const int row_block_size = bs->rows[r].block.size;
      const std::vector<Cell>& cells = bs->rows[r].cells;
      for (int c = 0; c < cells.size(); ++c) {
        const int col_block_id = cells[c].block_id;
        const int col_block_pos = bs->cols[col_block_id].position;
        const int col_block_size = bs->cols[col_block_id].size;
        MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
            values + cells[c].position,
            row_block_size,
            col_block_size,
            x + row_block_pos,
            y + col_block_pos - num_cols_e_);
      }
    }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyF_sequential_col(const double* x, double* y) const {
  const CompressedColumnBlockStructure* cs = matrix_.block_structure_columns();
  const double* values = matrix_.values();
  for (int c = num_col_blocks_e_; c < cs->cols.size(); ++c) {
    const int col_block_pos = cs->cols[c].block.position;
    const int col_block_size = cs->cols[c].block.size;
    for (auto& cell : cs->cols[c].cells) {
      const int row_block_id = cell.block_id;
      const int row_block_size = cs->rows[row_block_id].size;
      const int row_block_pos = cs->rows[row_block_id].position;

      if (row_block_id < num_row_blocks_e_) {
        MatrixTransposeVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
            values + cell.position,
            row_block_size,
            col_block_size,
            x + row_block_pos,
            y + col_block_pos - num_cols_e_);
      } else {
        MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
            values + cell.position,
            row_block_size,
            col_block_size,
            x + row_block_pos,
            y + col_block_pos - num_cols_e_);
      }
    }
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyF_parallel_col(const double* x, double* y) const {
  const CompressedColumnBlockStructure* cs = matrix_.block_structure_columns();
  const double* values = matrix_.values();
  ParallelFor(
      matrix_.GetContext(),
      num_col_blocks_e_,
      cs->cols.size(),
      matrix_.GetNumThreads(),
      [&](int c) {
        const int col_block_pos = cs->cols[c].block.position;
        const int col_block_size = cs->cols[c].block.size;
        for (auto& cell : cs->cols[c].cells) {
          const int row_block_id = cell.block_id;
          const int row_block_size = cs->rows[row_block_id].size;
          const int row_block_pos = cs->rows[row_block_id].position;

          if (row_block_id < num_row_blocks_e_) {
            MatrixTransposeVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
                values + cell.position,
                row_block_size,
                col_block_size,
                x + row_block_pos,
                y + col_block_pos - num_cols_e_);
          } else {
            MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
                values + cell.position,
                row_block_size,
                col_block_size,
                x + row_block_pos,
                y + col_block_pos - num_cols_e_);
          }
        }
      });
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyF_sequential_col_transpose(const double* x, double* y) const {
  const CompressedColumnBlockStructure* cs =
      matrix_.block_structure_columns_transpose();
  const double* values = matrix_.values_transpose();
  for (int c = num_col_blocks_e_; c < cs->cols.size(); ++c) {
    const int col_block_pos = cs->cols[c].block.position;
    const int col_block_size = cs->cols[c].block.size;
    for (auto& cell : cs->cols[c].cells) {
      const int row_block_id = cell.block_id;
      const int row_block_size = cs->rows[row_block_id].size;
      const int row_block_pos = cs->rows[row_block_id].position;

      if (row_block_id < num_row_blocks_e_) {
        MatrixTransposeVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
            values + cell.position,
            row_block_size,
            col_block_size,
            x + row_block_pos,
            y + col_block_pos - num_cols_e_);
      } else {
        MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
            values + cell.position,
            row_block_size,
            col_block_size,
            x + row_block_pos,
            y + col_block_pos - num_cols_e_);
      }
    }
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyF_parallel_col_transpose(const double* x, double* y) const {
  const CompressedColumnBlockStructure* cs =
      matrix_.block_structure_columns_transpose();
  const double* values = matrix_.values_transpose();
  ParallelFor(
      matrix_.GetContext(),
      num_col_blocks_e_,
      cs->cols.size(),
      matrix_.GetNumThreads(),
      [&](int c) {
        const int col_block_pos = cs->cols[c].block.position;
        const int col_block_size = cs->cols[c].block.size;
        for (auto& cell : cs->cols[c].cells) {
          const int row_block_id = cell.block_id;
          const int row_block_size = cs->rows[row_block_id].size;
          const int row_block_pos = cs->rows[row_block_id].position;

          if (row_block_id < num_row_blocks_e_) {
            MatrixTransposeVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
                values + cell.position,
                row_block_size,
                col_block_size,
                x + row_block_pos,
                y + col_block_pos - num_cols_e_);
          } else {
            MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
                values + cell.position,
                row_block_size,
                col_block_size,
                x + row_block_pos,
                y + col_block_pos - num_cols_e_);
          }
        }
      });
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyF_parallel_row(const double* x, double* y_) const {
  int threads = matrix_.GetNumThreads();
  size_t total = num_cols_f() * threads;
  std::unique_ptr<double[]> acc_(new double[total * threads]);
  double* acc = acc_.get();
  memset(acc, 0x0, sizeof(double) * total);
  memcpy(acc, y_, sizeof(double) * num_cols_f());
  const double* values = matrix_.values();

  const CompressedRowBlockStructure* bs = matrix_.block_structure();

  AutoTimer timer;
  ParallelFor(
      matrix_.GetContext(),
      0,
      bs->rows.size(),
      threads,
      [&](int thread_id, int r) {
        double* y = acc + thread_id * num_cols_f();
        if (r < num_row_blocks_e_) {
          const int row_block_pos = bs->rows[r].block.position;
          const int row_block_size = bs->rows[r].block.size;
          const std::vector<Cell>& cells = bs->rows[r].cells;
          for (int c = 1; c < cells.size(); ++c) {
            const int col_block_id = cells[c].block_id;
            const int col_block_pos = bs->cols[col_block_id].position;
            const int col_block_size = bs->cols[col_block_id].size;
            MatrixTransposeVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
                values + cells[c].position,
                row_block_size,
                col_block_size,
                x + row_block_pos,
                y + col_block_pos - num_cols_e_);
          }
        } else {
          const int row_block_pos = bs->rows[r].block.position;
          const int row_block_size = bs->rows[r].block.size;
          const std::vector<Cell>& cells = bs->rows[r].cells;
          for (int c = 0; c < cells.size(); ++c) {
            const int col_block_id = cells[c].block_id;
            const int col_block_pos = bs->cols[col_block_id].position;
            const int col_block_size = bs->cols[col_block_id].size;
            MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
                values + cells[c].position,
                row_block_size,
                col_block_size,
                x + row_block_pos,
                y + col_block_pos - num_cols_e_);
          }
        }
      });
  auto compute = timer.lap();
  aggregate_sum(threads, num_cols_f(), acc, matrix_.GetContext(), threads);
  memcpy(y_, acc, sizeof(double) * num_cols_f());
  auto aggregate = timer.lap();

  std::cout << "\nISB-A: " << __FUNCTION__ << ", " << matrix_.num_nonzeros()
            << ", " << num_rows() << ", " << num_cols() << ", " << num_cols_e()
            << ", " << num_cols_f() << ", " << compute.first << ", "
            << compute.second << std::endl;
  std::cout << "\nISB-A: " << __FUNCTION__ << ", "
            << ", " << matrix_.num_nonzeros() << ", " << num_rows() << ", "
            << num_cols() << ", " << num_cols_e() << ", " << num_cols_f()
            << ", " << aggregate.first << ", " << aggregate.second << std::endl;
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    LeftMultiplyF(const double* x, double* y) const {
  auto context = matrix_.GetContext();

  // Iterate over row blocks, and if the row block is in E, then
  // multiply by all the cells except the first one which is of type
  // E. If the row block is not in E (i.e its in the bottom
  // num_row_blocks - num_row_blocks_e row blocks), then all the cells
  // are of type F and multiply by them all.
  if (!context) {
    LeftMultiplyF_sequential_row(x, y);
  } else {
    std::unique_ptr<double[]> foo(new double[num_cols_f()]);
    memcpy(foo.get(), y, sizeof(double) * num_cols_f());
    void(matrix_.CreateTransposedStructure());

    BENCH_TIMER(LeftMultiplyF_sequential_row(x, foo.get()));
    BENCH_TIMER(LeftMultiplyF_sequential_row(x, foo.get()));
    BENCH_TIMER(LeftMultiplyF_sequential_col(x, foo.get()));
    BENCH_TIMER(LeftMultiplyF_sequential_col_transpose(x, foo.get()));
    BENCH_TIMER(LeftMultiplyF_parallel_col_transpose(x, foo.get()));
    BENCH_TIMER(LeftMultiplyF_parallel_col(x, y));
  }
}

// Given a range of columns blocks of a matrix m, compute the block
// structure of the block diagonal of the matrix m(:,
// start_col_block:end_col_block)'m(:, start_col_block:end_col_block)
// and return a BlockSparseMatrix with the this block structure. The
// caller owns the result.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
BlockSparseMatrix*
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
CreateBlockDiagonalMatrixLayout(int start_col_block, int end_col_block) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  CompressedRowBlockStructure* block_diagonal_structure =
      new CompressedRowBlockStructure;

  int block_position = 0;
  int diagonal_cell_position = 0;

  // Iterate over the column blocks, creating a new diagonal block for
  // each column block.
  for (int c = start_col_block; c < end_col_block; ++c) {
    const Block& block = bs->cols[c];
    block_diagonal_structure->cols.push_back(Block());
    Block& diagonal_block = block_diagonal_structure->cols.back();
    diagonal_block.size = block.size;
    diagonal_block.position = block_position;

    block_diagonal_structure->rows.push_back(CompressedRow());
    CompressedRow& row = block_diagonal_structure->rows.back();
    row.block = diagonal_block;

    row.cells.push_back(Cell());
    Cell& cell = row.cells.back();
    cell.block_id = c - start_col_block;
    cell.position = diagonal_cell_position;

    block_position += block.size;
    diagonal_cell_position += block.size * block.size;
  }

  // Build a BlockSparseMatrix with the just computed block
  // structure.
  return new BlockSparseMatrix(
      block_diagonal_structure, true, matrix_.GetContext());
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
BlockSparseMatrix*
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
CreateBlockDiagonalEtE() const {
  BlockSparseMatrix* block_diagonal =
      CreateBlockDiagonalMatrixLayout(0, num_col_blocks_e_);
  UpdateBlockDiagonalEtE(block_diagonal);
  return block_diagonal;
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
BlockSparseMatrix*
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
CreateBlockDiagonalFtF() const {
  BlockSparseMatrix* block_diagonal = CreateBlockDiagonalMatrixLayout(
      num_col_blocks_e_, num_col_blocks_e_ + num_col_blocks_f_);
  UpdateBlockDiagonalFtF(block_diagonal);
  return block_diagonal;
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalEtE_sequential_row(
        BlockSparseMatrix* block_diagonal) const {
  block_diagonal->SetZero();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const Cell& cell = bs->rows[r].cells[0];
    const int row_block_size = bs->rows[r].block.size;
    const int block_id = cell.block_id;
    const int col_block_size = bs->cols[block_id].size;
    const int cell_position =
        block_diagonal_structure->rows[block_id].cells[0].position;

    MatrixTransposeMatrixMultiply<kRowBlockSize,
                                  kEBlockSize,
                                  kRowBlockSize,
                                  kEBlockSize,
                                  1>(
        values + cell.position,
        row_block_size,
        col_block_size,
        values + cell.position,
        row_block_size,
        col_block_size,
        block_diagonal->mutable_values() + cell_position,
        0,
        0,
        col_block_size,
        col_block_size);
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalEtE_sequential_col(
        BlockSparseMatrix* block_diagonal) const {
  block_diagonal->SetZero();
  const CompressedColumnBlockStructure* cs = matrix_.block_structure_columns();
  const double* values = matrix_.values();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  for (int c = 0; c < num_col_blocks_e_; ++c) {
    const int col_block_pos = cs->cols[c].block.position;
    const int col_block_size = cs->cols[c].block.size;
    for (auto& cell : cs->cols[c].cells) {
      const int row_block_id = cell.block_id;
      const int row_block_size = cs->rows[row_block_id].size;
      const int row_block_pos = cs->rows[row_block_id].position;
      const int cell_position =
          block_diagonal_structure->rows[c].cells[0].position;

      MatrixTransposeMatrixMultiply<kRowBlockSize,
                                    kEBlockSize,
                                    kRowBlockSize,
                                    kEBlockSize,
                                    1>(
          values + cell.position,
          row_block_size,
          col_block_size,
          values + cell.position,
          row_block_size,
          col_block_size,
          block_diagonal->mutable_values() + cell_position,
          0,
          0,
          col_block_size,
          col_block_size);
    }
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalEtE_parallel_col(
        BlockSparseMatrix* block_diagonal) const {
  block_diagonal->SetZero();
  const CompressedColumnBlockStructure* cs = matrix_.block_structure_columns();
  const double* values = matrix_.values();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  ParallelFor(matrix_.GetContext(),
              0,
              num_col_blocks_e_,
              matrix_.GetNumThreads(),
              [&](int c) {
                const int col_block_pos = cs->cols[c].block.position;
                const int col_block_size = cs->cols[c].block.size;
                for (auto& cell : cs->cols[c].cells) {
                  const int row_block_id = cell.block_id;
                  const int row_block_size = cs->rows[row_block_id].size;
                  const int row_block_pos = cs->rows[row_block_id].position;
                  const int cell_position =
                      block_diagonal_structure->rows[c].cells[0].position;

                  MatrixTransposeMatrixMultiply<kRowBlockSize,
                                                kEBlockSize,
                                                kRowBlockSize,
                                                kEBlockSize,
                                                1>(
                      values + cell.position,
                      row_block_size,
                      col_block_size,
                      values + cell.position,
                      row_block_size,
                      col_block_size,
                      block_diagonal->mutable_values() + cell_position,
                      0,
                      0,
                      col_block_size,
                      col_block_size);
                }
              });
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalEtE_sequential_col_transpose(
        BlockSparseMatrix* block_diagonal) const {
  block_diagonal->SetZero();
  const CompressedColumnBlockStructure* cs =
      matrix_.block_structure_columns_transpose();
  const double* values = matrix_.values_transpose();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  for (int c = 0; c < num_col_blocks_e_; ++c) {
    const int col_block_pos = cs->cols[c].block.position;
    const int col_block_size = cs->cols[c].block.size;
    for (auto& cell : cs->cols[c].cells) {
      const int row_block_id = cell.block_id;
      const int row_block_size = cs->rows[row_block_id].size;
      const int row_block_pos = cs->rows[row_block_id].position;
      const int cell_position =
          block_diagonal_structure->rows[c].cells[0].position;

      MatrixTransposeMatrixMultiply<kRowBlockSize,
                                    kEBlockSize,
                                    kRowBlockSize,
                                    kEBlockSize,
                                    1>(
          values + cell.position,
          row_block_size,
          col_block_size,
          values + cell.position,
          row_block_size,
          col_block_size,
          block_diagonal->mutable_values() + cell_position,
          0,
          0,
          col_block_size,
          col_block_size);
    }
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalEtE_parallel_col_transpose(
        BlockSparseMatrix* block_diagonal) const {
  block_diagonal->SetZero();
  const CompressedColumnBlockStructure* cs =
      matrix_.block_structure_columns_transpose();
  const double* values = matrix_.values_transpose();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  ParallelFor(matrix_.GetContext(),
              0,
              num_col_blocks_e_,
              matrix_.GetNumThreads(),
              [&](int c) {
                const int col_block_pos = cs->cols[c].block.position;
                const int col_block_size = cs->cols[c].block.size;
                for (auto& cell : cs->cols[c].cells) {
                  const int row_block_id = cell.block_id;
                  const int row_block_size = cs->rows[row_block_id].size;
                  const int row_block_pos = cs->rows[row_block_id].position;
                  const int cell_position =
                      block_diagonal_structure->rows[c].cells[0].position;

                  MatrixTransposeMatrixMultiply<kRowBlockSize,
                                                kEBlockSize,
                                                kRowBlockSize,
                                                kEBlockSize,
                                                1>(
                      values + cell.position,
                      row_block_size,
                      col_block_size,
                      values + cell.position,
                      row_block_size,
                      col_block_size,
                      block_diagonal->mutable_values() + cell_position,
                      0,
                      0,
                      col_block_size,
                      col_block_size);
                }
              });
}

// Similar to the code in RightMultiplyE, except instead of the matrix
// vector multiply its an outer product.
//
//    block_diagonal = block_diagonal(E'E)
//
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalEtE(BlockSparseMatrix* block_diagonal) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();

  auto context = matrix_.GetContext();

  block_diagonal->SetZero();
  if (!context) {
    UpdateBlockDiagonalEtE_sequential_row(block_diagonal);
  } else {
    void(matrix_.CreateTransposedStructure());
    BENCH_TIMER(UpdateBlockDiagonalEtE_sequential_row(block_diagonal));
    BENCH_TIMER(UpdateBlockDiagonalEtE_sequential_col(block_diagonal));
    BENCH_TIMER(UpdateBlockDiagonalEtE_parallel_col(block_diagonal));
    BENCH_TIMER(UpdateBlockDiagonalEtE_parallel_col_transpose(block_diagonal));
    BENCH_TIMER(
        UpdateBlockDiagonalEtE_sequential_col_transpose(block_diagonal));
  }
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalFtF_sequential_row(
        BlockSparseMatrix* block_diagonal) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  block_diagonal->SetZero();
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const int row_block_size = bs->rows[r].block.size;
    const std::vector<Cell>& cells = bs->rows[r].cells;
    for (int c = 1; c < cells.size(); ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_size = bs->cols[col_block_id].size;
      const int diagonal_block_id = col_block_id - num_col_blocks_e_;
      const int cell_position =
          block_diagonal_structure->rows[diagonal_block_id].cells[0].position;

      MatrixTransposeMatrixMultiply<kRowBlockSize,
                                    kFBlockSize,
                                    kRowBlockSize,
                                    kFBlockSize,
                                    1>(
          values + cells[c].position,
          row_block_size,
          col_block_size,
          values + cells[c].position,
          row_block_size,
          col_block_size,
          block_diagonal->mutable_values() + cell_position,
          0,
          0,
          col_block_size,
          col_block_size);
    }
  }

  for (int r = num_row_blocks_e_; r < bs->rows.size(); ++r) {
    const int row_block_size = bs->rows[r].block.size;
    const std::vector<Cell>& cells = bs->rows[r].cells;
    for (int c = 0; c < cells.size(); ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_size = bs->cols[col_block_id].size;
      const int diagonal_block_id = col_block_id - num_col_blocks_e_;
      const int cell_position =
          block_diagonal_structure->rows[diagonal_block_id].cells[0].position;

      MatrixTransposeMatrixMultiply<Eigen::Dynamic,
                                    Eigen::Dynamic,
                                    Eigen::Dynamic,
                                    Eigen::Dynamic,
                                    1>(
          values + cells[c].position,
          row_block_size,
          col_block_size,
          values + cells[c].position,
          row_block_size,
          col_block_size,
          block_diagonal->mutable_values() + cell_position,
          0,
          0,
          col_block_size,
          col_block_size);
    }
  }
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalFtF_sequential_col(
        BlockSparseMatrix* block_diagonal) const {
  block_diagonal->SetZero();
  const CompressedColumnBlockStructure* cs = matrix_.block_structure_columns();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  const double* values = matrix_.values();

  for (int c = num_col_blocks_e_; c < cs->cols.size(); ++c) {
    const int col_block_pos = cs->cols[c].block.position;
    const int col_block_size = cs->cols[c].block.size;
    const int diagonal_block_id = c - num_col_blocks_e_;
    const int cell_position =
        block_diagonal_structure->rows[diagonal_block_id].cells[0].position;
    for (auto& cell : cs->cols[c].cells) {
      const int row_block_id = cell.block_id;
      const int row_block_size = cs->rows[row_block_id].size;
      const int row_block_pos = cs->rows[row_block_id].position;

      if (row_block_id < num_row_blocks_e_) {
        MatrixTransposeMatrixMultiply<kRowBlockSize,
                                      kFBlockSize,
                                      kRowBlockSize,
                                      kFBlockSize,
                                      1>(
            values + cell.position,
            row_block_size,
            col_block_size,
            values + cell.position,
            row_block_size,
            col_block_size,
            block_diagonal->mutable_values() + cell_position,
            0,
            0,
            col_block_size,
            col_block_size);
      } else {
        MatrixTransposeMatrixMultiply<Eigen::Dynamic,
                                      Eigen::Dynamic,
                                      Eigen::Dynamic,
                                      Eigen::Dynamic,
                                      1>(
            values + cell.position,
            row_block_size,
            col_block_size,
            values + cell.position,
            row_block_size,
            col_block_size,
            block_diagonal->mutable_values() + cell_position,
            0,
            0,
            col_block_size,
            col_block_size);
      }
    }
  };
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalFtF_parallel_col(
        BlockSparseMatrix* block_diagonal) const {
  block_diagonal->SetZero();
  const CompressedColumnBlockStructure* cs = matrix_.block_structure_columns();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  const double* values = matrix_.values();
  ParallelFor(
      matrix_.GetContext(),
      num_col_blocks_e_,
      cs->cols.size(),
      matrix_.GetNumThreads(),
      [&](int c) {
        const int col_block_pos = cs->cols[c].block.position;
        const int col_block_size = cs->cols[c].block.size;
        const int diagonal_block_id = c - num_col_blocks_e_;
        const int cell_position =
            block_diagonal_structure->rows[diagonal_block_id].cells[0].position;
        for (auto& cell : cs->cols[c].cells) {
          const int row_block_id = cell.block_id;
          const int row_block_size = cs->rows[row_block_id].size;
          const int row_block_pos = cs->rows[row_block_id].position;

          if (row_block_id < num_row_blocks_e_) {
            MatrixTransposeMatrixMultiply<kRowBlockSize,
                                          kFBlockSize,
                                          kRowBlockSize,
                                          kFBlockSize,
                                          1>(
                values + cell.position,
                row_block_size,
                col_block_size,
                values + cell.position,
                row_block_size,
                col_block_size,
                block_diagonal->mutable_values() + cell_position,
                0,
                0,
                col_block_size,
                col_block_size);
          } else {
            MatrixTransposeMatrixMultiply<Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          1>(
                values + cell.position,
                row_block_size,
                col_block_size,
                values + cell.position,
                row_block_size,
                col_block_size,
                block_diagonal->mutable_values() + cell_position,
                0,
                0,
                col_block_size,
                col_block_size);
          }
        }
      });
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalFtF_sequential_col_transpose(
        BlockSparseMatrix* block_diagonal) const {
  block_diagonal->SetZero();
  const CompressedColumnBlockStructure* cs =
      matrix_.block_structure_columns_transpose();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  const double* values = matrix_.values_transpose();

  for (int c = num_col_blocks_e_; c < cs->cols.size(); ++c) {
    const int col_block_pos = cs->cols[c].block.position;
    const int col_block_size = cs->cols[c].block.size;
    const int diagonal_block_id = c - num_col_blocks_e_;
    const int cell_position =
        block_diagonal_structure->rows[diagonal_block_id].cells[0].position;
    for (auto& cell : cs->cols[c].cells) {
      const int row_block_id = cell.block_id;
      const int row_block_size = cs->rows[row_block_id].size;
      const int row_block_pos = cs->rows[row_block_id].position;

      if (row_block_id < num_row_blocks_e_) {
        MatrixTransposeMatrixMultiply<kRowBlockSize,
                                      kFBlockSize,
                                      kRowBlockSize,
                                      kFBlockSize,
                                      1>(
            values + cell.position,
            row_block_size,
            col_block_size,
            values + cell.position,
            row_block_size,
            col_block_size,
            block_diagonal->mutable_values() + cell_position,
            0,
            0,
            col_block_size,
            col_block_size);
      } else {
        MatrixTransposeMatrixMultiply<Eigen::Dynamic,
                                      Eigen::Dynamic,
                                      Eigen::Dynamic,
                                      Eigen::Dynamic,
                                      1>(
            values + cell.position,
            row_block_size,
            col_block_size,
            values + cell.position,
            row_block_size,
            col_block_size,
            block_diagonal->mutable_values() + cell_position,
            0,
            0,
            col_block_size,
            col_block_size);
      }
    }
  };
}
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalFtF_parallel_col_transpose(
        BlockSparseMatrix* block_diagonal) const {
  block_diagonal->SetZero();
  const CompressedColumnBlockStructure* cs =
      matrix_.block_structure_columns_transpose();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  const double* values = matrix_.values_transpose();
  ParallelFor(
      matrix_.GetContext(),
      num_col_blocks_e_,
      cs->cols.size(),
      matrix_.GetNumThreads(),
      [&](int c) {
        const int col_block_pos = cs->cols[c].block.position;
        const int col_block_size = cs->cols[c].block.size;
        const int diagonal_block_id = c - num_col_blocks_e_;
        const int cell_position =
            block_diagonal_structure->rows[diagonal_block_id].cells[0].position;
        for (auto& cell : cs->cols[c].cells) {
          const int row_block_id = cell.block_id;
          const int row_block_size = cs->rows[row_block_id].size;
          const int row_block_pos = cs->rows[row_block_id].position;

          if (row_block_id < num_row_blocks_e_) {
            MatrixTransposeMatrixMultiply<kRowBlockSize,
                                          kFBlockSize,
                                          kRowBlockSize,
                                          kFBlockSize,
                                          1>(
                values + cell.position,
                row_block_size,
                col_block_size,
                values + cell.position,
                row_block_size,
                col_block_size,
                block_diagonal->mutable_values() + cell_position,
                0,
                0,
                col_block_size,
                col_block_size);
          } else {
            MatrixTransposeMatrixMultiply<Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          1>(
                values + cell.position,
                row_block_size,
                col_block_size,
                values + cell.position,
                row_block_size,
                col_block_size,
                block_diagonal->mutable_values() + cell_position,
                0,
                0,
                col_block_size,
                col_block_size);
          }
        }
      });
}

// Similar to the code in RightMultiplyF, except instead of the matrix
// vector multiply its an outer product.
//
//   block_diagonal = block_diagonal(F'F)
//
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
    UpdateBlockDiagonalFtF(BlockSparseMatrix* block_diagonal) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();

  auto context = matrix_.GetContext();
  if (!context) {
    UpdateBlockDiagonalFtF_sequential_row(block_diagonal);
  } else {
    void(matrix_.CreateTransposedStructure());

    BENCH_TIMER(UpdateBlockDiagonalFtF_sequential_row(block_diagonal));
    BENCH_TIMER(UpdateBlockDiagonalFtF_sequential_col(block_diagonal));
    BENCH_TIMER(
        UpdateBlockDiagonalFtF_sequential_col_transpose(block_diagonal));
    BENCH_TIMER(UpdateBlockDiagonalFtF_parallel_col_transpose(block_diagonal));
    BENCH_TIMER(UpdateBlockDiagonalFtF_parallel_col(block_diagonal));
  }
}

}  // namespace internal
}  // namespace ceres
