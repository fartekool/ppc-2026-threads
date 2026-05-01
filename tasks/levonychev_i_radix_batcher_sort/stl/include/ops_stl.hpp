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
  static void RadixSortSequential(std::vector<int>& arr);
  static void MergeAndSplit(std::vector<int>& left_block, std::vector<int>& right_block);
};

}  // namespace levonychev_i_radix_batcher_sort
