#pragma once

#include <cstddef>
#include <vector>

#include "levonychev_i_radix_batcher_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace levonychev_i_radix_batcher_sort {

class LevonychevIRadixBatcherSortSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }
  explicit LevonychevIRadixBatcherSortSTL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void CountingSort(std::vector<int>& arr, std::vector<int>& buffer, size_t byte_index);
  static void BatcherCompareRange(std::vector<int> &arr, int j, int k, int p2);
};

}  // namespace levonychev_i_radix_batcher_sort
