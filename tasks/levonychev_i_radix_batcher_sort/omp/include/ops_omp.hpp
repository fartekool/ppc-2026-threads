#pragma once

#include "levonychev_i_radix_batcher_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace levonychev_i_radix_batcher_sort {

class LevonychevIRadixBatcherSortOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit LevonychevIRadixBatcherSortOMP(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void CountingSort(InType& arr, size_t byte_index);
  inline static void CompareExchange(int& a, int& b);
  void BatcherMergeIterative(std::vector<int>& arr);
};

}  // namespace levonychev_i_radix_batcher_sort
