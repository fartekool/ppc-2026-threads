#include "levonychev_i_radix_batcher_sort/stl/include/ops_stl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <future>
#include <iterator>
#include <thread>
#include <vector>

#include "levonychev_i_radix_batcher_sort/common/include/common.hpp"
namespace levonychev_i_radix_batcher_sort {

LevonychevIRadixBatcherSortSTL::LevonychevIRadixBatcherSortSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

void LevonychevIRadixBatcherSortSTL::RadixSortSequential(std::vector<int> &arr) {
  if (arr.empty()) {
    return;
  }
  int n = static_cast<int>(arr.size());
  std::vector<int> buffer(n);

  for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {
    std::array<int, 256> count{};
    bool is_last_byte = (byte_idx == 3);

    for (int x : arr) {
      unsigned char b = (static_cast<unsigned int>(x) >> (byte_idx * 8)) & 0xFF;
      if (is_last_byte) {
        b ^= 0x80;
      }
      count[b]++;
    }

    int offsets[256];
    offsets[0] = 0;
    for (int i = 1; i < 256; ++i) {
      offsets[i] = offsets[i - 1] + count[i - 1];
    }

    for (int x : arr) {
      unsigned char b = (static_cast<unsigned int>(x) >> (byte_idx * 8)) & 0xFF;
      if (is_last_byte) {
        b ^= 0x80;
      }
      buffer[offsets[b]++] = x;
    }
    arr = buffer;
  }
}

void LevonychevIRadixBatcherSortSTL::MergeAndSplit(std::vector<int> &left_block, std::vector<int> &right_block) {
  std::vector<int> merged;
  merged.reserve(left_block.size() + right_block.size());

  std::ranges::merge(left_block, right_block, std::back_inserter(merged));

  auto mid = static_cast<std::ptrdiff_t>(left_block.size());
  std::copy(merged.begin(), merged.begin() + mid, left_block.begin());
  std::copy(merged.begin() + mid, merged.end(), right_block.begin());
}

bool LevonychevIRadixBatcherSortSTL::RunImpl() {
  std::vector<int> data = GetInput();
  int n = static_cast<int>(data.size());
  if (n <= 1) {
    GetOutput() = data;
    return true;
  }

  int num_blocks = 1;
  unsigned int threads_supported = std::thread::hardware_concurrency();
  int max_threads = static_cast<int>(threads_supported == 0 ? 2 : threads_supported);
  while (num_blocks * 2 <= max_threads) {
    num_blocks *= 2;
  }

  std::vector<std::vector<int>> blocks(num_blocks);
  int base_size = n / num_blocks;
  int extra = n % num_blocks;

  int current_pos = 0;
  for (int i = 0; i < num_blocks; ++i) {
    int size = base_size + (i < extra ? 1 : 0);
    blocks[i].assign(data.begin() + current_pos, data.begin() + current_pos + size);
    current_pos += size;
  }
  std::vector<std::future<void>> futures;
  for (int i = 0; i < num_blocks; ++i) {
    futures.push_back(std::async(std::launch::async, [&blocks, i]() { RadixSortSequential(blocks[i]); }));
  }
  for (auto &f : futures) {
    f.wait();
  }
  futures.clear();
  for (int p = 1; p < num_blocks; p <<= 1) {
    for (int k = p; k > 0; k >>= 1) {
      for (int j = k % p; j <= num_blocks - 1 - k; j += 2 * k) {
        futures.reserve(static_cast<size_t>(num_blocks));
        for (int i = 0; i < std::min(k, num_blocks - j - k); ++i) {
          if ((j + i) / (p * 2) == (j + i + k) / (p * 2)) {
            futures.push_back(std::async(std::launch::async, [&blocks, idx1 = j + i, idx2 = j + i + k]() {
              MergeAndSplit(blocks[idx1], blocks[idx2]);
            }));
          }
        }
        for (auto &f : futures) {
          f.wait();
        }
        futures.clear();
      }
    }
  }
  GetOutput().clear();
  GetOutput().reserve(n);
  for (const auto &b : blocks) {
    GetOutput().insert(GetOutput().end(), b.begin(), b.end());
  }

  return true;
}

bool LevonychevIRadixBatcherSortSTL::ValidationImpl() {
  return !GetInput().empty();
}
bool LevonychevIRadixBatcherSortSTL::PreProcessingImpl() {
  return true;
}
bool LevonychevIRadixBatcherSortSTL::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_radix_batcher_sort
