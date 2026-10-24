// Derived from ggml's AVX2 ggml_vec_dot_q4_K_q8_K at
// f2918cabbffe8abf8e2a90c1c089085fa116c2cf, arch/x86/quants.c.
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "q4_pair.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <cassert>
#include <cstring>

namespace wllm::experiment {
static float reduce(__m256 acc, __m128 minimum) {
    minimum = _mm_add_ps(minimum, _mm_movehl_ps(minimum, minimum));
    minimum = _mm_add_ss(minimum, _mm_movehdup_ps(minimum));
    __m128 sum = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum) + _mm_cvtss_f32(minimum);
}

void q4_pair(int n, float * first, float * second, const void * weights,
             const void * activation_first, const void * activation_second) {
    assert(n > 0 && n % QK_K == 0);
    const auto * x = static_cast<const block_q4_K *>(weights);
    const auto * y0 = static_cast<const block_q8_K *>(activation_first);
    const auto * y1 = static_cast<const block_q8_K *>(activation_second);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m128 min0 = _mm_setzero_ps(), min1 = _mm_setzero_ps();
    const auto mask = _mm256_set1_epi8(15);
    for (int i = 0; i < n / QK_K; ++i) {
        // Decode scale/min once, but do not regroup either dot product.
        const float dx = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[i].data.data.d)));
        const float mx = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[i].data.data.dmin)));
        uint32_t u[4];
        std::memcpy(u, x[i].scales, 12);
        u[3] = ((u[2] >> 4) & 0x0f0f0f0f) | (((u[1] >> 6) & 0x03030303) << 4);
        const uint32_t a = u[1] & 0x3f3f3f3f;
        u[1] = (u[2] & 0x0f0f0f0f) | (((u[0] >> 6) & 0x03030303) << 4);
        u[2] = a; u[0] &= 0x3f3f3f3f;
        const __m256i scales_mins = _mm256_cvtepu8_epi16(_mm_set_epi32(u[3],u[2],u[1],u[0]));
        const __m128i mins = _mm256_extracti128_si256(scales_mins,1);
        const __m128i scales128 = _mm256_castsi256_si128(scales_mins);
        const __m256i scales = _mm256_inserti128_si256(_mm256_castsi128_si256(scales128),scales128,1);
        auto minimum = [&](const block_q8_K & y, __m128 current) {
            const auto sums = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y.bsums));
            const auto sums32 = _mm_hadd_epi16(_mm256_castsi256_si128(sums),_mm256_extracti128_si256(sums,1));
            const auto prod = _mm_madd_epi16(mins,sums32);
            return _mm_fmadd_ps(_mm_set1_ps(-y.d * mx),_mm_cvtepi32_ps(prod),current);
        };
        min0 = minimum(y0[i],min0); min1 = minimum(y1[i],min1);
        __m256i sum0 = _mm256_setzero_si256(), sum1 = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 64; ++j) {
            const auto lo = _mm256_shuffle_epi8(scales,_mm256_set1_epi16(short(0x0100 + 0x0404*j)));
            const auto hi = _mm256_shuffle_epi8(scales,_mm256_set1_epi16(short(0x0302 + 0x0404*j)));
            const auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x[i].qs + 32*j));
            const auto ql = _mm256_and_si256(bits,mask);
            const auto qh = _mm256_and_si256(_mm256_srli_epi16(bits,4),mask);
            auto partial = [&](const block_q8_K & y) {
                const auto yl = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y.qs + 64*j));
                const auto yh = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y.qs + 64*j+32));
                return _mm256_add_epi32(_mm256_madd_epi16(lo,_mm256_maddubs_epi16(ql,yl)),
                                       _mm256_madd_epi16(hi,_mm256_maddubs_epi16(qh,yh)));
            };
            sum0 = _mm256_add_epi32(sum0,partial(y0[i]));
            sum1 = _mm256_add_epi32(sum1,partial(y1[i]));
        }
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(y0[i].d * dx),_mm256_cvtepi32_ps(sum0),acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(y1[i].d * dx),_mm256_cvtepi32_ps(sum1),acc1);
    }
    *first = reduce(acc0,min0); *second = reduce(acc1,min1);
}
}
