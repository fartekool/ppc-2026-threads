#include "levonychev_i_radix_batcher_sort/stl/include/ops_stl.hpp"
#include <algorithm>
#include <barrier>
#include <vector>
#include <thread>
#include <cmath>

namespace levonychev_i_radix_batcher_sort {

LevonychevIRadixBatcherSortSTL::LevonychevIRadixBatcherSortSTL(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
}

void LevonychevIRadixBatcherSortSTL::CountingSort(std::vector<int>& arr, std::vector<int>& buffer, size_t byte_index) {
    const size_t radix = 256;
    size_t count[radix] = {0};

    bool is_last_byte = (byte_index == (sizeof(int) - 1));
    for (int val : arr) {
        unsigned char b = static_cast<unsigned char>((static_cast<unsigned int>(val) >> (byte_index * 8)) & 0xFF);
        if (is_last_byte) b ^= 0x80;
        count[b]++;
    }

    size_t offsets[radix];
    offsets[0] = 0;
    for (size_t i = 1; i < radix; ++i) offsets[i] = offsets[i - 1] + count[i - 1];

    for (int val : arr) {
        unsigned char b = static_cast<unsigned char>((static_cast<unsigned int>(val) >> (byte_index * 8)) & 0xFF);
        if (is_last_byte) b ^= 0x80;
        buffer[offsets[b]++] = val;
    }
    arr = buffer;
}

void LevonychevIRadixBatcherSortSTL::BatcherCompareRange(std::vector<int> &arr, int j, int k, int p2) {
    int n = static_cast<int>(arr.size());
    int range = std::min(k, n - j - k);
    for (int i = 0; i < range; ++i) {
        int idx1 = j + i;
        int idx2 = j + i + k;
        if ((idx1 & p2) == (idx2 & p2) && (arr[idx1] > arr[idx2])) {
            std::swap(arr[idx1], arr[idx2]);
        }
    }
}

bool LevonychevIRadixBatcherSortSTL::RunImpl() {
    GetOutput() = GetInput();
    int n = static_cast<int>(GetOutput().size());
    if (n <= 1) return true;

    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    num_threads = 1;
    if (num_threads < 1) num_threads = 1;

    int block_size = n / num_threads;
    int start_p = 1;
    while (start_p < block_size) start_p <<= 1;

    std::barrier sync_point(num_threads);

    auto worker = [&](int tid) {
        int left = tid * block_size;
        int right = (tid == num_threads - 1) ? n : (tid + 1) * block_size;

        if (left < right) {
            std::vector<int> local_block(GetOutput().begin() + left, GetOutput().begin() + right);
            std::vector<int> buffer(local_block.size());
            for (size_t i = 0; i < sizeof(int); ++i) {
                CountingSort(local_block, buffer, i);
            }
            std::copy(local_block.begin(), local_block.end(), GetOutput().begin() + left);
        }

        sync_point.arrive_and_wait();

        for (int pv = start_p; pv < n; pv <<= 1) {
            int p2 = pv << 1;
            for (int k = pv; k > 0; k >>= 1) {
                int start_j = k % pv;
                int step = 2 * k;
                for (int j = start_j + tid * step; j < n - k; j += num_threads * step) {
                    BatcherCompareRange(GetOutput(), j, k, p2);
                }
                
                sync_point.arrive_and_wait();
            }
        }
    };

    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back(worker, i);
    }

    for (auto &t : workers) {
        t.join();
    }

    return true;
}

bool LevonychevIRadixBatcherSortSTL::ValidationImpl() { return !GetInput().empty(); }
bool LevonychevIRadixBatcherSortSTL::PreProcessingImpl() { return true; }
bool LevonychevIRadixBatcherSortSTL::PostProcessingImpl() { return true; }

} // namespace levonychev_i_radix_batcher_sort