#include "levonychev_i_radix_batcher_sort/omp/include/ops_omp.hpp"

#include <atomic>
#include <numeric>
#include <vector>
#include <omp.h>
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

inline void LevonychevIRadixBatcherSortOMP::CompareExchange(int& a, int& b) {
    if (a > b) {
        std::swap(a, b);
    }
}



void LevonychevIRadixBatcherSortOMP::BatcherMergeIterative(std::vector<int>& arr) {
    int n = static_cast<int>(arr.size());
    if (n < 2) return;

    for (int p = 1; p < n; p <<= 1) {
        for (int k = p; k > 0; k >>= 1) {
            #pragma omp parallel
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
    InType& data = GetInput();

    int n = static_cast<int>(data.size());
    GetOutput() = data; 
    OutType& res = GetOutput();

    if (n == 1) return true;

    int num_threads = omp_get_max_threads();
    int block_size = (n + num_threads - 1) / num_threads;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int left = tid * block_size;
        int right = std::min(left + block_size, n);

        if (left < right) {
            std::vector<int> local_block(res.begin() + left, res.begin() + right);
            
            for (size_t byte_idx = 0; byte_idx < sizeof(int); ++byte_idx) {
                CountingSort(local_block, byte_idx);
            }
            std::copy(local_block.begin(), local_block.end(), res.begin() + left);
        }
    }
    BatcherMergeIterative(res);

    return true;
}

bool LevonychevIRadixBatcherSortOMP::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_radix_batcher_sort
