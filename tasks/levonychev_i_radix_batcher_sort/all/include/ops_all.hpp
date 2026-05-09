#pragma once

#include <vector>
#include <mpi.h>
#include "levonychev_i_radix_batcher_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace levonychev_i_radix_batcher_sort {

class LevonychevIRadixBatcherSortALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }
  explicit LevonychevIRadixBatcherSortALL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void LocalRadixSort(std::vector<int>& arr);
  static void LocalParallelMerge(std::vector<std::vector<int>>& blocks);
  
  // Сетевые методы (MPI)
  static void NetworkMergeAndSplit(std::vector<int>& local_data, int partner, bool keep_low);
};

}  // namespace levonychev_i_radix_batcher_sort
