#include "levonychev_i_radix_batcher_sort/omp/include/ops_omp.hpp"

#include <omp.h>

#include <atomic>
#include <numeric>
#include <vector>

#include "levonychev_i_radix_batcher_sort/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_radix_batcher_sort {

LevonychevIRadixBatcherSortOMP::LevonychevIRadixBatcherSortOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

void LevonychevIRadixBatcherSortOMP::CountingSort(InType &arr, size_t byte_index) {
  const size_t byte = 256;
  std::vector<int> count(byte, 0);
  OutType result(arr.size());

  bool is_last_byte = (byte_index == (sizeof(int) - 1ULL));
  for (auto number : arr) {
    int value_of_byte = (number >> (byte_index * 8ULL)) & 0xFF;

    if (is_last_byte) {
      value_of_byte ^= 0x80;
    }

    ++count[value_of_byte];
  }

  for (size_t i = 1ULL; i < byte; ++i) {
    count[i] += count[i - 1];
  }

  for (int &val : std::ranges::reverse_view(arr)) {
    int value_of_byte = (val >> (byte_index * 8ULL)) & 0xFF;

    if (is_last_byte) {
      value_of_byte ^= 0x80;
    }

    result[--count[value_of_byte]] = val;
  }
  arr = result;
}

bool LevonychevIRadixBatcherSortOMP::ValidationImpl() {
  return !GetInput().empty();
}

bool LevonychevIRadixBatcherSortOMP::PreProcessingImpl() {
  return true;
}

inline void LevonychevIRadixBatcherSortOMP::CompareExchange(int &a, int &b) {
  if (a > b) {
    std::swap(a, b);
  }
}

void LevonychevIRadixBatcherSortOMP::BatcherMergeIterative(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  if (n < 2) {
    return;
  }
  // int tnum = omp_get_max_threads();

  for (int p = 1; p < n; p <<= 1) {
    for (int k = p; k > 0; k >>= 1) {
#pragma omp parallel num_threads(8)
      {
#pragma omp for schedule(static)
        for (int j = k % p; j < n - k; j += 2 * k) {
          int range = std::min(k, n - j - k);
          for (int i = 0; i < range; ++i) {
            int idx1 = j + i;
            int idx2 = j + i + k;
            if (idx1 / (p * 2) == idx2 / (p * 2)) {
              if (arr[idx1] > arr[idx2]) {
                std::swap(arr[idx1], arr[idx2]);
              }
            }
          }
        }
      }
    }
  }
}

bool LevonychevIRadixBatcherSortOMP::RunImpl() {
  GetOutput() = GetInput();
  int n = static_cast<int>(GetOutput().size());
  if (n <= 1) {
    return true;
  }

  int num_threads = omp_get_max_threads();
  int block_size = n / num_threads;

// 1. Параллельная локальная сортировка (Radix)
#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    int left = tid * block_size;
    int right = (tid == num_threads - 1) ? n : (tid + 1) * block_size;

    if (left < right) {
      std::vector<int> local_block(GetOutput().begin() + left, GetOutput().begin() + right);
      for (size_t i = 0; i < sizeof(int); ++i) {
        CountingSort(local_block, i);
      }
      std::copy(local_block.begin(), local_block.end(), GetOutput().begin() + left);
    }
  }

  // 2. Находим стартовое 'p'
  // Если наши блоки размера block_size уже отсортированы,
  // мы можем начать слияние сразу с p >= block_size.
  int start_p = 1;
  while (start_p < block_size) {
    start_p <<= 1;
  }

  // 3. Запускаем модифицированный итеративный Бетчер
  for (int p = 1; p < n; p <<= 1) {
    int p2 = p << 1;
    for (int k = p; k > 0; k >>= 1) {
#pragma omp parallel for schedule(static)
      for (int j = k % p; j < n - k; j += 2 * k) {
        int range = std::min(k, n - j - k);
        for (int i = 0; i < range; ++i) {
          int idx1 = j + i;
          int idx2 = j + i + k;
          if ((idx1 & p2) == (idx2 & p2)) {
            if (GetOutput()[idx1] > GetOutput()[idx2]) {
              std::swap(GetOutput()[idx1], GetOutput()[idx2]);
            }
          }
        }
      }
    }
  }

  return true;
}

bool LevonychevIRadixBatcherSortOMP::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_radix_batcher_sort
