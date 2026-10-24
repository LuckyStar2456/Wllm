#pragma once
#include <cstddef>

namespace wllm::experiment {
// Research-only AVX2/FMA/F16C kernel. Two activation columns share Q4_K
// unpacking; each column keeps the upstream accumulation/reduction order.
// n must be positive and divisible by 256. Inputs use pinned ggml layouts.
void q4_pair(int n, float * first, float * second, const void * weights,
             const void * activation_first, const void * activation_second);
}
