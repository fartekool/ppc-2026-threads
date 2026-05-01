#include "levonychev_i_radix_batcher_sort/stl/include/ops_stl.hpp"

#include <cstddef>
#include <ranges>
#include <vector>

#include "levonychev_i_radix_batcher_sort/common/include/common.hpp"

namespace levonychev_i_radix_batcher_sort {

LevonychevIRadixBatcherSortSTL::LevonychevIRadixBatcherSortSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

void LevonychevIRadixBatcherSortSTL::CountingSort(InType &arr, size_t byte_index) {
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

bool LevonychevIRadixBatcherSortSTL::ValidationImpl() {
  return !GetInput().empty();
}

bool LevonychevIRadixBatcherSortSTL::PreProcessingImpl() {
  return true;
}

bool LevonychevIRadixBatcherSortSTL::RunImpl() {
  GetOutput() = GetInput();

  for (size_t i = 0; i < sizeof(int); ++i) {
    CountingSort(GetOutput(), i);
  }

  return true;
}

bool LevonychevIRadixBatcherSortSTL::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_radix_batcher_sort
