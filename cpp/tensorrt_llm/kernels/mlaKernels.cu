/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

// A stateful callback functor that maintains the running sum between consecutive scans.
struct BlockPrefixCallbackOp
{
    // Running prefix
    int mRunningTotal;

    // Constructor
    __device__ BlockPrefixCallbackOp(int runningTotal)
        : mRunningTotal(runningTotal)
    {
    }

    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int blockAggregate)
    {
        int oldPrefix = mRunningTotal;
        mRunningTotal += blockAggregate;
        return oldPrefix;
    }
};

template <typename T>
struct VecType
{
    using Type = T;
};

template <>
struct VecType<float>
{
    using Type = float4;
};

template <>
struct VecType<half>
{
    using Type = uint4;
};

template <>
struct VecType<__nv_bfloat16>
{
    using Type = mmha::bf16_8_t;
};

template <typename T>
struct loadPagedKVKernelTraits
{
    static constexpr int kLoraSize = 512;
    static constexpr int kRopeSize = 64;
    static constexpr int kHeadSize = kLoraSize + kRopeSize;
    using VecT = typename VecType<T>::Type;
    static constexpr int kBytesPerElem = sizeof(T);
    static constexpr int kBytesPerLoad = 16;
    static constexpr int kElemPerLoad = kBytesPerLoad / kBytesPerElem;
    static_assert((kHeadSize * kBytesPerElem) % kBytesPerLoad == 0,
        "kHeadSize * kBytesPerElem must be multiple of kBytesPerLoad (16Bytes)");
    static constexpr int kVecPerHead = (kHeadSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kThreadPerHead = kVecPerHead; // for each head, we use kThreadPerHead threads to fetch all the
                                                       // kv cache data, each thread read kv cache only once.
    static constexpr int kTokenPerBlock
        = std::is_same_v<T, float> ? 4 : 8;            // for each block, we fetch 8 token for fp16, 4 tokens for fp32.
    static constexpr int kBlockSize = kThreadPerHead * kTokenPerBlock;
    static constexpr int kKVThreadPerHead = (kLoraSize * kBytesPerElem) / kBytesPerLoad;
};

template <typename T>
struct setPagedKVKernelTraits
{
    static constexpr int kQKNopeSize = 128;
    static constexpr int kVHeadSize = 128;
    static_assert(kQKNopeSize == kVHeadSize);
    static constexpr int kRopeSize = 64;
    static constexpr int kHeadSize = kQKNopeSize + kRopeSize;
    using VecT = typename VecType<T>::Type;
    static constexpr int kBytesPerElem = sizeof(T);
    static constexpr int kBytesPerLoad = 16;
    static constexpr int kElemPerLoad = kBytesPerLoad / kBytesPerElem;
    static_assert((kHeadSize * kBytesPerElem) % kBytesPerLoad == 0,
        "kHeadSize * kBytesPerElem must be multiple of kBytesPerLoad (16Bytes)");
    static constexpr int kNumHeads = 128;
    static constexpr int kThreadPerHead = (kHeadSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kKVThreadPerHead = (kQKNopeSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kCpTokenPerBlock = 16;
    static constexpr int kBlockSize = kThreadPerHead * kCpTokenPerBlock;
};

namespace mla
{

template <typename T>
inline __device__ void apply_rotary_embedding_mla(
    T& q, T q_pair_left, T q_pair_right, T& k, T k_pair_left, T k_pair_right, float2 const& coef)
{
    T cos = cuda_cast<T>(coef.x);
    T sin = cuda_cast<T>(coef.y);

    q = cuda_cast<T>(cuda_cast<float>(cos * q_pair_left)) + cuda_cast<T>(cuda_cast<float>(sin * q_pair_right));
    k = cuda_cast<T>(cuda_cast<float>(cos * k_pair_left)) + cuda_cast<T>(cuda_cast<float>(sin * k_pair_right));
}

template <typename T>
inline __device__ void apply_rotary_embedding_mla(T& q, T q_left, T q_right, float2 const& coef)
{
    T cos = cuda_cast<T>(coef.x);
    T sin = cuda_cast<T>(coef.y);

    q = cuda_cast<T>(cuda_cast<float>(cos * q_left)) + cuda_cast<T>(cuda_cast<float>(sin * q_right));
}

} // namespace mla

template <typename SrcType, int NUM>
inline __device__ void quantCopy(
    __nv_fp8_e4m3* dst_global_ptr, SrcType const* src_fragment_ptr, float const scale_val = 1.f)
{
    using DstVecType = typename std::conditional<sizeof(SrcType) == 2, float2, float>::type;
    using SrcType2 =
        typename std::conditional<sizeof(SrcType) == 2, typename TypeConverter<SrcType>::Type, float2>::type;
    static constexpr int COPY_SIZE = sizeof(DstVecType);
    static constexpr int TOTAL_COPY_SIZE = NUM * sizeof(__nv_fp8_e4m3);
    static constexpr int LOOP_NUM = TOTAL_COPY_SIZE / COPY_SIZE;
    static_assert(TOTAL_COPY_SIZE % COPY_SIZE == 0);
    static constexpr int CVT_NUM = COPY_SIZE / sizeof(__nv_fp8_e4m3) / 2;
    static_assert(COPY_SIZE % (sizeof(__nv_fp8_e4m3) * 2) == 0);
    DstVecType fragment;
#pragma unroll
    for (int i = 0; i < LOOP_NUM; ++i)
    {
#pragma unroll
        for (int j = 0; j < CVT_NUM; ++j)
        {
            float2 val2 = cuda_cast<float2>(reinterpret_cast<SrcType2 const*>(src_fragment_ptr)[j]);
            val2.x *= scale_val;
            val2.y *= scale_val;
            reinterpret_cast<__nv_fp8x2_e4m3*>(&fragment)[j] = __nv_fp8x2_e4m3(val2);
        }
        reinterpret_cast<DstVecType*>(dst_global_ptr)[i] = fragment;
    }
}

template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer>
__global__ void applyMLARopeAndAssignQKVKernelOptContext(T* qkv_output, T const* fuse_buf, KVCacheBuffer kv_cache,
    float2 const* cos_sin_cache, size_t head_num, int head_size, int c_k, int* cu_q_seqlens,
    int32_t const* kv_cache_lengths, uint32_t max_input_seq_len, KvCacheDataType cache_type,
    float const* quant_scale_kv)
{

    // Constants.
    using VecT = typename VecType<T>::Type;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto HALF_ROTATARY_DIM = ROPE_DIM / 2;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VECS_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;

    // Block/Head idx.
    size_t const batch_idx = blockIdx.y;
    size_t const head_idx = blockIdx.z;

    if (head_idx < head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
        bool const first_half = head_dim_idx < HALF_ROTATARY_DIM;

        size_t const seq_len_loop_end
            = size_t((max_input_seq_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {

            int const global_token_offset = cu_q_seqlens[batch_idx];
            int const cache_seq_len = kv_cache_lengths[batch_idx];
            int token_idx_in_kv_cache = local_token_idx;
            bool const valid_token = token_idx_in_kv_cache < cache_seq_len;
            // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
            token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
            local_token_idx = std::min(local_token_idx, cache_seq_len - 1);
            int const global_token_idx = local_token_idx + global_token_offset;

            auto const position_id = local_token_idx;
            auto const src_bias = first_half ? head_dim_idx * 2 : (head_dim_idx - HALF_ROTATARY_DIM) * 2;
            float2 const* rotary_coef_cache_buffer
                = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx);

            VecT q, k;
            VecT q_ref[2], k_ref[2];
            auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM) + c_k;
            auto const src_q_global_offset
                = static_cast<size_t>(global_token_idx) * head_num * ((head_size + ROPE_DIM) * 2 + head_size)
                + (head_size + ROPE_DIM) * head_idx + head_size;

            for (int i = 0; i < 2; ++i)
            {
                q_ref[i]
                    = *reinterpret_cast<VecT const*>(&qkv_output[src_q_global_offset + src_bias + i * ELTS_PER_VEC]);
                k_ref[i] = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + src_bias + i * ELTS_PER_VEC]);
            }

            for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
            {
                float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                rotary_coef_cache.y = first_half ? -rotary_coef_cache.y : rotary_coef_cache.y;
                auto& q_ = reinterpret_cast<T*>(&q)[elt_id];
                auto& k_ = reinterpret_cast<T*>(&k)[elt_id];
                auto q_left = first_half ? reinterpret_cast<T*>(&q_ref)[elt_id * 2]
                                         : reinterpret_cast<T*>(&q_ref)[elt_id * 2 + 1];
                auto q_right = first_half ? reinterpret_cast<T*>(&q_ref)[elt_id * 2 + 1]
                                          : reinterpret_cast<T*>(&q_ref)[elt_id * 2];
                auto k_left = first_half ? reinterpret_cast<T*>(&k_ref)[elt_id * 2]
                                         : reinterpret_cast<T*>(&k_ref)[elt_id * 2 + 1];
                auto k_right = first_half ? reinterpret_cast<T*>(&k_ref)[elt_id * 2 + 1]
                                          : reinterpret_cast<T*>(&k_ref)[elt_id * 2];
                // float2 rotary_coef_cache;
                // T q_left, q_right, k_left, k_right;
                mla::apply_rotary_embedding_mla(q_, q_left, q_right, k_, k_left, k_right, rotary_coef_cache);
            }
            // do sync
            __syncwarp();
            if (valid_token)
            {
                if (head_idx == 0)
                {
                    auto kDst = reinterpret_cast<T*>(kv_cache.getVBlockPtr(batch_idx, token_idx_in_kv_cache));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(
                        token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                    if (cache_type == KvCacheDataType::FP8)
                    {

                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * 8,
                            reinterpret_cast<T const*>(&k), quant_scale_kv_val);
                    }
                    else
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx] = k;
                }
                if (head_idx == 0)
                {
                    auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(
                        token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                    if (cache_type == KvCacheDataType::FP8)
                    {

                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * 8,
                            reinterpret_cast<T const*>(&k), quant_scale_kv_val);
                    }
                    else
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx] = k;
                }
                auto const dst_q_idx
                    = static_cast<size_t>(global_token_idx) * head_num * ((head_size + ROPE_DIM) * 2 + head_size)
                    + head_idx * (head_size + ROPE_DIM) + head_size + head_dim_idx;
                auto const dst_k_idx
                    = static_cast<size_t>(global_token_idx) * head_num * ((head_size + ROPE_DIM) * 2 + head_size)
                    + head_num * (head_size + ROPE_DIM) + head_idx * (head_size + ROPE_DIM) + head_size + head_dim_idx;
                reinterpret_cast<VecT*>(qkv_output)[dst_q_idx / ELTS_PER_VEC] = q;
                reinterpret_cast<VecT*>(qkv_output)[dst_k_idx / ELTS_PER_VEC] = k;
            }
        }
    }
    else
    {
        int block_dim = gridDim.z - head_num;
        int block_id = head_idx - head_num;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_seq_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {

            int const global_token_offset = cu_q_seqlens[batch_idx];
            int const cache_seq_len = kv_cache_lengths[batch_idx];
            int token_idx_in_kv_cache = local_token_idx;
            bool const valid_token = token_idx_in_kv_cache < cache_seq_len;
            // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
            token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
            local_token_idx = std::min(local_token_idx, cache_seq_len - 1);
            int const global_token_idx = local_token_idx + global_token_offset;

            if (valid_token)
            {
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM);

                auto kDst = reinterpret_cast<T*>(kv_cache.getVBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto inBlockIdx
                    = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, head_dim_vec_idx);
                if (cache_type == KvCacheDataType::FP8)
                {

                    quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * 8,
                        fuse_buf + src_k_global_offset + head_dim_idx, quant_scale_kv_val);
                }
                else
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                        = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);
            }
            if (valid_token)
            {
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM);

                auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto inBlockIdx
                    = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, head_dim_vec_idx);
                if (cache_type == KvCacheDataType::FP8)
                {

                    quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * 8,
                        fuse_buf + src_k_global_offset + head_dim_idx, quant_scale_kv_val);
                }
                else
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                        = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);
            }
        }
    }
}

template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer>
__global__ void applyMLARopeAndAssignQKVKernelGeneration(T* qkv_output, T* q_pe, T const* fuse_buf, void* quant_q,
    KVCacheBuffer kv_cache, float2 const* cos_sin_cache, size_t head_num, int c_k, int total_s_len, int seq_len,
    int* seqQOffset, uint32_t* fmha_tile_counter, int32_t const* kv_cache_lengths, int* seqKVOffsets, int q_pe_ld,
    int q_pe_stride, KvCacheDataType cache_type, float* bmm1_scale, float* bmm2_scale, float const* quant_scale_o,
    float const* quant_scale_q, float const* quant_scale_kv, float const* dequant_scale_q,
    float const* dequant_scale_kv, float host_bmm1_scale)
{

    // Constants.
    using VecT = typename VecType<T>::Type;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto HALF_ROTATARY_DIM = ROPE_DIM / 2;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VEC_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;

    // Block/Head idx.
    size_t const head_idx = blockIdx.y;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
        fmha_tile_counter[0] = 0;
        seqQOffset[0] = 0;

        // Calculate bmm scale for FP8 MLA
        if (cache_type == KvCacheDataType::FP8)
        {
            float dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
            float dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
            float quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;
            if (bmm1_scale)
            {
                // The scale prepared for log2 optimization.
                constexpr float kLog2e = 1.4426950408889634074f;
                // The scale after fmha bmm1.
                float bmm1_scale_val = dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
                bmm1_scale[0] = bmm1_scale_val;
                bmm1_scale[1] = bmm1_scale_val * kLog2e;
            }
            if (bmm2_scale)
            {
                // The scale after fmha bmm2.
                bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
            }
        }
    }

    if (head_idx <= head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
        bool const first_half = head_dim_idx < HALF_ROTATARY_DIM;

        int const seq_len_loop_end = size_t((total_s_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float const quant_scale_q_val = quant_scale_q ? quant_scale_q[0] : 1.0f;
        float const quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.0f;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {
            auto batch_idx = global_token_idx / seq_len;
            auto local_token_idx = global_token_idx % seq_len;
            bool const valid_token = global_token_idx < total_s_len;
            VecT data;

            if (valid_token)
            {
                VecT ref[2];

                auto const position_id = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;
                auto const src_bias = first_half ? head_dim_idx * 2 : (head_dim_idx - HALF_ROTATARY_DIM) * 2;
                float2 const* rotary_coef_cache_buffer
                    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx);

                if (head_idx == head_num)
                {
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM) + c_k;

                    for (int i = 0; i < 2; ++i)
                    {
                        ref[i] = *reinterpret_cast<VecT const*>(
                            &fuse_buf[src_k_global_offset + src_bias + i * ELTS_PER_VEC]);
                    }
                }
                else
                {
                    auto const src_q_global_offset
                        = static_cast<size_t>(global_token_idx) * q_pe_stride + q_pe_ld * head_idx;

                    for (int i = 0; i < 2; ++i)
                    {
                        ref[i]
                            = *reinterpret_cast<VecT const*>(&q_pe[src_q_global_offset + src_bias + i * ELTS_PER_VEC]);
                    }
                }

                for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
                {
                    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                    rotary_coef_cache.y = first_half ? -rotary_coef_cache.y : rotary_coef_cache.y;
                    auto& data_ = reinterpret_cast<T*>(&data)[elt_id];
                    auto data_left = first_half ? reinterpret_cast<T*>(&ref)[elt_id * 2]
                                                : reinterpret_cast<T*>(&ref)[elt_id * 2 + 1];
                    auto data_right = first_half ? reinterpret_cast<T*>(&ref)[elt_id * 2 + 1]
                                                 : reinterpret_cast<T*>(&ref)[elt_id * 2];
                    mla::apply_rotary_embedding_mla(data_, data_left, data_right, rotary_coef_cache);
                }
            }

            __syncwarp();

            if (valid_token)
            {
                if (head_idx == head_num)
                {
                    auto const token_kv_idx = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;

                    {
                        auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                        auto inBlockIdx = kv_cache.getKVLocalIdx(
                            token_kv_idx, 0, TOTAL_VEC_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                        if (cache_type == KvCacheDataType::FP8)
                        {

                            quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * 8,
                                reinterpret_cast<T const*>(&data), quant_scale_kv_val);
                        }
                        else
                            reinterpret_cast<VecT*>(kDst)[inBlockIdx] = data;
                    }
                }
                else
                {
                    auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (c_k + ROPE_DIM)
                        + head_idx * (c_k + ROPE_DIM) + c_k + head_dim_idx;
                    if (cache_type == KvCacheDataType::FP8)
                    {
                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(quant_q) + dst_q_idx,
                            reinterpret_cast<T const*>(&data), quant_scale_q_val);
                    }
                    else
                        reinterpret_cast<VecT*>(qkv_output)[dst_q_idx / ELTS_PER_VEC] = data;
                }
            }
        }
    }
    else if (head_idx <= head_num + 8)
    {
        int block_dim = gridDim.y - head_num - 1;
        int block_id = head_idx - head_num - 1;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((total_s_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.0f;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {
            auto batch_idx = global_token_idx / seq_len;
            auto local_token_idx = global_token_idx % seq_len;
            bool valid_token = global_token_idx < total_s_len;

            if (valid_token)
            {
                if (head_dim_vec_idx == 0)
                {
                    seqQOffset[batch_idx + 1] = head_num * seq_len * (batch_idx + 1);
                }

                auto const token_kv_idx = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;
                auto const src_kv_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM);

                {
                    auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(token_kv_idx, 0, TOTAL_VEC_PER_HEAD, head_dim_vec_idx);

                    if (cache_type == KvCacheDataType::FP8)
                    {
                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * 8,
                            fuse_buf + src_kv_global_offset + head_dim_idx, quant_scale_kv_val);
                    }
                    else
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                            = *reinterpret_cast<VecT const*>(&fuse_buf[src_kv_global_offset + head_dim_idx]);
                }
            }
        }
    }
    else
    {
        if (cache_type == KvCacheDataType::FP8)
        {
            int block_dim = gridDim.y - head_num - 1 - 8;
            int block_id = head_idx - head_num - 1 - 8;
            size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
            size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
            size_t const head_num_idx = (block_id % head_num) * (K_HEAD_SIZE + HEAD_SIZE);

            size_t const seq_len_loop_end
                = size_t((total_s_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
            float quant_scale_q_val = quant_scale_q ? quant_scale_q[0] : 1.0f;

            // Mainloop.
            for (int global_token_idx = (threadIdx.x / K_VECS_PER_HEAD)
                     + (block_id / head_num) * gridDim.x * K_TOKENS_PER_BLOCK + blockIdx.x * K_TOKENS_PER_BLOCK;
                 global_token_idx < seq_len_loop_end;
                 global_token_idx += (block_dim / head_num) * gridDim.x * K_TOKENS_PER_BLOCK)
            {
                if (global_token_idx < total_s_len)
                {
                    size_t const load_idx
                        = global_token_idx * head_num * (K_HEAD_SIZE + HEAD_SIZE) + head_num_idx + head_dim_idx;
                    quantCopy<T, ELTS_PER_VEC>(
                        reinterpret_cast<__nv_fp8_e4m3*>(quant_q) + load_idx, qkv_output + load_idx, quant_scale_q_val);
                }
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif

    // The implementation of the parallel scan in the thread block (see CUB for details).
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;

    // Allocate storage in shared memory to do the scan.
    __shared__ typename BlockScan::TempStorage tempKVStorage;
    BlockPrefixCallbackOp prefixKVOp(0);

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        int const batchSizeBound = total_s_len / seq_len;
        for (int batchOffset = 0; batchOffset <= batchSizeBound; batchOffset += BLOCK_SIZE)
        {
            // The index of the batch.
            int batchIdx = batchOffset + threadIdx.x;
            int seqKVLength = 0;
            if (batchIdx < batchSizeBound)
            {
                seqKVLength = kv_cache_lengths[batchIdx];
            }
            int seqKVOffset;
            BlockScan(tempKVStorage).ExclusiveSum(seqKVLength, seqKVOffset, prefixKVOp);
            if (batchIdx <= batchSizeBound)
            {
                seqKVOffsets[batchIdx] = seqKVOffset;
            }
        }
    }
}

template <typename T>
__global__ void loadPagedKVCacheForMLAKernel(T* kv_output, const tensorrt_llm::kernels::KVBlockArray kv_cache,
    int64_t const* cu_ctx_cached_kv_lens, int max_input_seq_len)
{
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<T>;

    int const batch_idx = static_cast<int>(blockIdx.y);
    int const head_idx = static_cast<int>(blockIdx.z);

    size_t const head_dim_vec_idx = (threadIdx.x % KT::kVecPerHead);
    size_t const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;

    size_t const seq_len_loop_end
        = (max_input_seq_len + KT::kTokenPerBlock - 1) / KT::kTokenPerBlock * KT::kTokenPerBlock;

    int64_t const global_token_offset = cu_ctx_cached_kv_lens[batch_idx];
    int64_t const cache_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];

    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kTokenPerBlock;
         local_token_idx < seq_len_loop_end; local_token_idx += KT::kTokenPerBlock * gridDim.x)
    {
        int token_idx_in_kv_cache = local_token_idx;
        bool const valid_token = token_idx_in_kv_cache < cache_kv_len;

        if (valid_token)
        {
            auto* kvSrc = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
            // head_idx === 0
            auto kvBlockIdx
                = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, KT::kVecPerHead, static_cast<int>(head_dim_vec_idx));

            // kv_output {total_token, num_heads, head_size}
            int const global_token_idx = local_token_idx + global_token_offset;
            int const dstIdx = global_token_idx * gridDim.z * KT::kHeadSize + head_idx * KT::kHeadSize + head_dim_idx;

            // copy back to kv_output
            *reinterpret_cast<typename KT::VecT*>(kv_output + dstIdx)
                = reinterpret_cast<typename KT::VecT*>(kvSrc)[kvBlockIdx];
        }
    }
}

// k {total_token, h, d}, v {total_token, h, d}, k_pe {total_token, h=1, d_rope}
// output {b, 2, ceil(max_seq / kv_cache_tokens_per_block), h, kv_cache_tokens_per_block, d}
template <typename T>
__global__ void setPagedKVCacheForMLAKernel(T* output, T* const k_ptr, T* const v_ptr, T* const k_pe_ptr,
    int64_t const* cu_seq_lens, int const max_input_seq_len, int num_heads, int kv_dim, int rope_dim,
    int kv_cache_tokens_per_block)
{
    using KT = typename tensorrt_llm::kernels::setPagedKVKernelTraits<T>;
    int const batch_idx = static_cast<int>(blockIdx.y);
    int const head_idx = static_cast<int>(blockIdx.z);
    int const head_dim_vec_idx = (threadIdx.x % KT::kThreadPerHead);
    int const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;
    bool const is_valid_v = head_dim_idx < KT::kVHeadSize;

    size_t const seq_len_loop_end
        = (max_input_seq_len + KT::kCpTokenPerBlock - 1) / KT::kCpTokenPerBlock * KT::kCpTokenPerBlock;
    size_t const kv_cache_block_size = num_heads * kv_cache_tokens_per_block * (kv_dim + rope_dim);
    size_t const kv_cache_block_num = (max_input_seq_len + kv_cache_tokens_per_block - 1) / kv_cache_tokens_per_block;
    int64_t const global_token_offset = cu_seq_lens[batch_idx];
    int64_t const cache_kv_len = cu_seq_lens[batch_idx + 1] - cu_seq_lens[batch_idx];

    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kCpTokenPerBlock;
         local_token_idx < seq_len_loop_end; local_token_idx += KT::kCpTokenPerBlock * gridDim.x)
    {
        int token_idx_in_kv_cache = local_token_idx;
        bool const valid_token = token_idx_in_kv_cache < cache_kv_len;
        if (valid_token)
        {
            // copy k and v
            if (is_valid_v)
            {
                int ld_kv_global_offset
                    = (global_token_offset + local_token_idx) * num_heads * kv_dim + head_idx * kv_dim;
                int ld_kv_local_offset = head_dim_vec_idx;
                auto k_data = (reinterpret_cast<typename KT::VecT*>(k_ptr + ld_kv_global_offset))[ld_kv_local_offset];
                auto v_data = (reinterpret_cast<typename KT::VecT*>(v_ptr + ld_kv_global_offset))[ld_kv_local_offset];
                // {b, 0, token / kv_cache_tokens_per_block, h, token % kv_cache_tokens_per_block, ...}
                int st_k_global_offset = batch_idx * 2 * kv_cache_block_num * kv_cache_block_size
                    + local_token_idx / kv_cache_tokens_per_block * kv_cache_block_size
                    + +head_idx * kv_cache_tokens_per_block * (kv_dim + rope_dim)
                    + (local_token_idx % kv_cache_tokens_per_block) * (kv_dim + rope_dim);
                // {b, 1, token / kv_cache_tokens_per_block, h, token % kv_cache_tokens_per_block, ...}
                int st_v_global_offset = st_k_global_offset + kv_cache_block_num * kv_cache_block_size;
                int st_k_local_offset = head_dim_vec_idx;
                int st_v_local_offset = head_dim_vec_idx;
                (reinterpret_cast<typename KT::VecT*>(output + st_k_global_offset))[st_k_local_offset] = k_data;
                (reinterpret_cast<typename KT::VecT*>(output + st_v_global_offset))[st_v_local_offset] = v_data;
            }
            // copy k_pe, only 1 head
            else
            {
                int ld_rope_global_offset = (global_token_offset + local_token_idx) * rope_dim;
                int ld_rope_local_offset = head_dim_vec_idx - KT::kKVThreadPerHead;
                auto rope_data
                    = (reinterpret_cast<typename KT::VecT*>(k_pe_ptr + ld_rope_global_offset))[ld_rope_local_offset];
                // {b, 0, token / kv_cache_tokens_per_block, h, token % kv_cache_tokens_per_block, ...}
                int st_rope_global_offset = batch_idx * 2 * kv_cache_block_num * kv_cache_block_size
                    + local_token_idx / kv_cache_tokens_per_block * kv_cache_block_size
                    + head_idx * kv_cache_tokens_per_block * (kv_dim + rope_dim)
                    + (local_token_idx % kv_cache_tokens_per_block) * (kv_dim + rope_dim);
                int st_rope_local_offset = head_dim_vec_idx;
                (reinterpret_cast<typename KT::VecT*>(output + st_rope_global_offset))[st_rope_local_offset]
                    = rope_data;
            }
        }
        else
        {
            break;
        }
    }
}

// ck {total_cached_token, h, d}, cv {total_cached_token, h, d}, ck_pe {total_cached_token, d_rope}
// nk {total_new_token, h, d}, nv {total_new_token, h, d}, nk_pe {total_new_token, d_rope}
// output {b, 2, ceil(max_seq / kv_cache_tokens_per_block), h, kv_cache_tokens_per_block, d}
template <typename T>
__global__ void setPagedKVCacheForMLAKernelV2(T* output, T* const cached_k_ptr, T* const cached_v_ptr,
    T* const cached_k_pe_ptr, T* const new_k_ptr, T* const new_v_ptr, T* const new_k_pe_ptr,
    int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens, int const max_input_seq_len, int num_heads,
    int kv_dim, int rope_dim, int kv_cache_tokens_per_block)
{
    using KT = typename tensorrt_llm::kernels::setPagedKVKernelTraits<T>;
    int const batch_idx = static_cast<int>(blockIdx.y);
    int const head_idx = static_cast<int>(blockIdx.z);
    int const head_dim_vec_idx = (threadIdx.x % KT::kThreadPerHead);
    int const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;
    bool const is_valid_v = head_dim_idx < KT::kVHeadSize;

    size_t const seq_len_loop_end
        = (max_input_seq_len + KT::kCpTokenPerBlock - 1) / KT::kCpTokenPerBlock * KT::kCpTokenPerBlock;
    size_t const kv_cache_block_size = num_heads * kv_cache_tokens_per_block * (kv_dim + rope_dim);
    size_t const kv_cache_block_num = (max_input_seq_len + kv_cache_tokens_per_block - 1) / kv_cache_tokens_per_block;
    int64_t const cached_global_token_offset = cu_ctx_cached_kv_lens[batch_idx];
    int64_t const uncached_global_token_offset = cu_seq_lens[batch_idx] - cu_ctx_cached_kv_lens[batch_idx];
    int64_t const total_kv_len = cu_seq_lens[batch_idx + 1] - cu_seq_lens[batch_idx];
    int64_t const cached_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];

    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kCpTokenPerBlock;
         local_token_idx < seq_len_loop_end; local_token_idx += KT::kCpTokenPerBlock * gridDim.x)
    {
        int token_idx_in_kv_cache = local_token_idx;
        bool const valid_token = token_idx_in_kv_cache < total_kv_len;
        if (valid_token)
        {
            // copy k and v
            if (is_valid_v)
            {
                int ld_kv_global_offset = local_token_idx < cached_kv_len
                    ? ((cached_global_token_offset + local_token_idx) * num_heads * kv_dim + head_idx * kv_dim)
                    : ((uncached_global_token_offset + local_token_idx - cached_kv_len) * num_heads * kv_dim
                        + head_idx * kv_dim);
                int ld_kv_local_offset = head_dim_vec_idx;
                auto k_ptr = local_token_idx < cached_kv_len ? cached_k_ptr : new_k_ptr;
                auto v_ptr = local_token_idx < cached_kv_len ? cached_v_ptr : new_v_ptr;
                auto k_data = (reinterpret_cast<typename KT::VecT*>(k_ptr + ld_kv_global_offset))[ld_kv_local_offset];
                auto v_data = (reinterpret_cast<typename KT::VecT*>(v_ptr + ld_kv_global_offset))[ld_kv_local_offset];
                // {b, 0, token / kv_cache_tokens_per_block, h, token % kv_cache_tokens_per_block, ...}
                int st_k_global_offset = batch_idx * 2 * kv_cache_block_num * kv_cache_block_size
                    + local_token_idx / kv_cache_tokens_per_block * kv_cache_block_size
                    + +head_idx * kv_cache_tokens_per_block * (kv_dim + rope_dim)
                    + (local_token_idx % kv_cache_tokens_per_block) * (kv_dim + rope_dim);
                // {b, 1, token / kv_cache_tokens_per_block, h, token % kv_cache_tokens_per_block, ...}
                int st_v_global_offset = st_k_global_offset + kv_cache_block_num * kv_cache_block_size;
                int st_k_local_offset = head_dim_vec_idx;
                int st_v_local_offset = head_dim_vec_idx;
                (reinterpret_cast<typename KT::VecT*>(output + st_k_global_offset))[st_k_local_offset] = k_data;
                (reinterpret_cast<typename KT::VecT*>(output + st_v_global_offset))[st_v_local_offset] = v_data;
            }
            // copy k_pe, only 1 head
            else
            {
                int ld_rope_global_offset = local_token_idx < cached_kv_len
                    ? ((cached_global_token_offset + local_token_idx) * rope_dim)
                    : ((uncached_global_token_offset + local_token_idx - cached_kv_len) * rope_dim);
                int ld_rope_local_offset = head_dim_vec_idx - KT::kKVThreadPerHead;
                auto k_pe_ptr = local_token_idx < cached_kv_len ? cached_k_pe_ptr : new_k_pe_ptr;
                auto rope_data
                    = (reinterpret_cast<typename KT::VecT*>(k_pe_ptr + ld_rope_global_offset))[ld_rope_local_offset];
                // {b, 0, token / kv_cache_tokens_per_block, h, token % kv_cache_tokens_per_block, ...}
                int st_rope_global_offset = batch_idx * 2 * kv_cache_block_num * kv_cache_block_size
                    + local_token_idx / kv_cache_tokens_per_block * kv_cache_block_size
                    + head_idx * kv_cache_tokens_per_block * (kv_dim + rope_dim)
                    + (local_token_idx % kv_cache_tokens_per_block) * (kv_dim + rope_dim);
                int st_rope_local_offset = head_dim_vec_idx;
                (reinterpret_cast<typename KT::VecT*>(output + st_rope_global_offset))[st_rope_local_offset]
                    = rope_data;
            }
        }
        else
        {
            break;
        }
    }
}

// compressed_kv_ptr {total_uncached_tokens, d}, k_pe_ptr {total_uncached_tokens, d_rope}
template <typename T>
__global__ void setCompressedPagedKVForMLAKernel(KVBlockArray kv_cache, T* const compressed_kv_ptr, T* const k_pe_ptr,
    int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens, int const max_input_uncached_seq_len,
    int head_dim)
{
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<T>;

    int const batch_idx = static_cast<int>(blockIdx.y);

    size_t const head_dim_vec_idx = (threadIdx.x % KT::kVecPerHead);
    size_t const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;
    bool const is_valid_kv = head_dim_vec_idx < KT::kKVThreadPerHead;

    size_t const seq_len_loop_end
        = (max_input_uncached_seq_len + KT::kTokenPerBlock - 1) / KT::kTokenPerBlock * KT::kTokenPerBlock;

    int64_t const global_token_offset = cu_seq_lens[batch_idx] - cu_ctx_cached_kv_lens[batch_idx];
    int64_t const cached_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];
    int64_t const uncached_kv_len = cu_seq_lens[batch_idx + 1] - cu_seq_lens[batch_idx] - cached_kv_len;

    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kTokenPerBlock;
         local_token_idx < seq_len_loop_end; local_token_idx += KT::kTokenPerBlock * gridDim.x)
    {
        int token_idx_in_kv_cache = local_token_idx + cached_kv_len;
        bool valid_token = local_token_idx < uncached_kv_len;
        if (valid_token)
        {
            typename KT::VecT src_data;
            if (is_valid_kv)
            {
                int ld_kv_global_offset = (global_token_offset + local_token_idx) * KT::kLoraSize;
                int ld_kv_local_offset = head_dim_vec_idx;
                src_data = (reinterpret_cast<typename KT::VecT*>(
                    compressed_kv_ptr + ld_kv_global_offset))[ld_kv_local_offset];
            }
            else
            {
                int ld_rope_global_offset = (global_token_offset + local_token_idx) * KT::kRopeSize;
                int ld_rope_local_offset = head_dim_vec_idx - KT::kKVThreadPerHead;
                src_data
                    = (reinterpret_cast<typename KT::VecT*>(k_pe_ptr + ld_rope_global_offset))[ld_rope_local_offset];
            }
            auto* kvCacheDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
            auto kvBlockIdx
                = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, KT::kVecPerHead, static_cast<int>(head_dim_vec_idx));
            reinterpret_cast<typename KT::VecT*>(kvCacheDst)[kvBlockIdx] = src_data;
        }
    }
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(params.max_input_seq_len, 32)), params.batch_size, params.head_num + 8);
    auto head_size = params.meta.qk_nope_head_dim;
    applyMLARopeAndAssignQKVKernelOptContext<T, 256, 512, 64, KVCacheBuffer>
        <<<grid, 256, 0, stream>>>(params.attention_input_buf, params.latent_cache, kv_cache_buffer,
            params.cos_sin_cache, params.head_num, head_size, params.meta.kv_lora_rank, params.cu_q_seqlens,
            params.cache_seq_lens, params.max_input_seq_len, params.cache_type, params.quant_scale_kv);
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(params.acc_q_len, 32)), params.head_num + 1 + 8);
    if (params.cache_type == KvCacheDataType::FP8)
        grid.y += params.head_num * 8;
    TLLM_CHECK_WITH_INFO(params.acc_q_len % params.batch_size == 0,
        "MLA can only support input sequences with the same sequence length.");
    auto seq_len = params.acc_q_len / params.batch_size;

    auto* kernel_instance = &applyMLARopeAndAssignQKVKernelGeneration<T, 256, 512, 64, KVCacheBuffer>;
    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = 256;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel_instance, params.attention_input_buf, params.q_pe, params.latent_cache,
        params.quant_attention_input_buf, kv_cache_buffer, params.cos_sin_cache, params.head_num,
        params.meta.kv_lora_rank, params.acc_q_len, seq_len, params.seqQOffset, params.fmha_tile_counter,
        params.cache_seq_lens, params.cu_kv_seqlens, params.q_pe_ld, params.q_pe_stride, params.cache_type,
        params.bmm1_scale, params.bmm2_scale, params.quant_scale_o, params.quant_scale_q, params.quant_scale_kv,
        params.dequant_scale_q, params.dequant_scale_kv, params.host_bmm1_scale);
}

template <typename T>
void invokeLoadPagedKVKernel(T* kv_output, KVBlockArray& kv_cache, int const num_contexts,
    int64_t const* cu_ctx_cached_kv_lens, int const max_input_seq_len, int head_dim, cudaStream_t stream)
{
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<T>;
    // {seq_len / token_per_block, batch_size, head_num}
    TLLM_CHECK_WITH_INFO(head_dim == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    dim3 grid(static_cast<int>(tensorrt_llm::common::divUp(max_input_seq_len, KT::kTokenPerBlock)), num_contexts, 1);
    loadPagedKVCacheForMLAKernel<T>
        <<<grid, KT::kBlockSize, 0, stream>>>(kv_output, kv_cache, cu_ctx_cached_kv_lens, max_input_seq_len);
}

template <typename T>
void invokeSetPagedKVKernel(T* output, T* const k_ptr, T* const v_ptr, T* const k_pe_ptr, int const num_requests,
    int64_t const* cu_seq_lens, int const max_input_seq_len, int num_heads, int kv_dim, int rope_dim,
    int kv_cache_tokens_per_block, cudaStream_t stream)
{
    using KT = typename tensorrt_llm::kernels::setPagedKVKernelTraits<T>;
    TLLM_CHECK_WITH_INFO(kv_dim + rope_dim == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    TLLM_CHECK_WITH_INFO(kv_cache_tokens_per_block % KT::kCpTokenPerBlock == 0,
        "kv_cache_tokens_per_block should be multiple of %d", KT::kCpTokenPerBlock);
    dim3 grid(tensorrt_llm::common::divUp(max_input_seq_len, KT::kCpTokenPerBlock), num_requests, num_heads);
    setPagedKVCacheForMLAKernel<T><<<grid, KT::kBlockSize, 0, stream>>>(output, k_ptr, v_ptr, k_pe_ptr, cu_seq_lens,
        max_input_seq_len, num_heads, kv_dim, rope_dim, kv_cache_tokens_per_block);
}

template <typename T>
void invokeSetPagedKVKernelV2(T* output, T* const cached_k_ptr, T* const cached_v_ptr, T* const cached_k_pe_ptr,
    T* const new_k_ptr, T* const new_v_ptr, T* const new_k_pe_ptr, int const num_requests,
    int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens, int const max_input_seq_len, int num_heads,
    int kv_dim, int rope_dim, int kv_cache_tokens_per_block, cudaStream_t stream)
{
    using KT = typename tensorrt_llm::kernels::setPagedKVKernelTraits<T>;
    TLLM_CHECK_WITH_INFO(kv_dim + rope_dim == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    TLLM_CHECK_WITH_INFO(kv_cache_tokens_per_block % KT::kCpTokenPerBlock == 0,
        "kv_cache_tokens_per_block should be multiple of %d", KT::kCpTokenPerBlock);
    dim3 grid(tensorrt_llm::common::divUp(max_input_seq_len, KT::kCpTokenPerBlock), num_requests, num_heads);
    setPagedKVCacheForMLAKernelV2<T><<<grid, KT::kBlockSize, 0, stream>>>(output, cached_k_ptr, cached_v_ptr,
        cached_k_pe_ptr, new_k_ptr, new_v_ptr, new_k_pe_ptr, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_seq_len,
        num_heads, kv_dim, rope_dim, kv_cache_tokens_per_block);
}

template <typename T>
void invokeSetCompressedPagedKVKernel(KVBlockArray& kv_cache, T* const compressed_kv_ptr, T* const k_pe_ptr,
    int const num_requests, int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens,
    int const max_input_uncached_seq_len, int head_dim, cudaStream_t stream)
{
    // just reuse the same traits as loadPagedKVKernel.
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<T>;
    TLLM_CHECK_WITH_INFO(head_dim == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    dim3 grid(
        static_cast<int>(tensorrt_llm::common::divUp(max_input_uncached_seq_len, KT::kTokenPerBlock)), num_requests, 1);
    setCompressedPagedKVForMLAKernel<T><<<grid, KT::kBlockSize, 0, stream>>>(kv_cache, compressed_kv_ptr, k_pe_ptr,
        cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, head_dim);
}

#define INSTANTIATE_MLA_ROPE(T, KVCacheBuffer)                                                                         \
    template void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);      \
    template void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

INSTANTIATE_MLA_ROPE(float, KVBlockArray);
INSTANTIATE_MLA_ROPE(half, KVBlockArray);
INSTANTIATE_MLA_ROPE(float, KVLinearBuffer);
INSTANTIATE_MLA_ROPE(half, KVLinearBuffer);

#ifdef ENABLE_BF16
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVBlockArray);
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVLinearBuffer);
#endif

#define INSTANTIATE_LOAD_KVCACHE_MLA(T)                                                                                \
    template void invokeLoadPagedKVKernel(T* kv_output, KVBlockArray& kv_cache, const int num_contexts,                \
        const int64_t* cu_ctx_cached_kv_lens, const int max_input_seq_len, int head_dim, cudaStream_t stream);         \
    template void invokeSetPagedKVKernel(T* output, T* const k_ptr, T* const v_ptr, T* const k_pe_ptr,                 \
        int const num_requests, int64_t const* cu_seq_lens, int const max_input_seq_len, int num_heads, int kv_dim,    \
        int rope_dim, int kv_cache_tokens_per_block, cudaStream_t stream);                                             \
    template void invokeSetPagedKVKernelV2(T* output, T* const cached_k_ptr, T* const cached_v_ptr,                    \
        T* const cached_k_pe_ptr, T* const new_k_ptr, T* const new_v_ptr, T* const new_k_pe_ptr,                       \
        int const num_requests, int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens,                      \
        int const max_input_seq_len, int num_heads, int kv_dim, int rope_dim, int kv_cache_tokens_per_block,           \
        cudaStream_t stream);                                                                                          \
    template void invokeSetCompressedPagedKVKernel(KVBlockArray& kv_cache, T* const compressed_kv_ptr,                 \
        T* const k_pe_ptr, int const num_requests, int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens,   \
        int const max_input_uncached_seq_len, int head_dim, cudaStream_t stream);

INSTANTIATE_LOAD_KVCACHE_MLA(float);
INSTANTIATE_LOAD_KVCACHE_MLA(half);
#ifdef ENABLE_BF16
INSTANTIATE_LOAD_KVCACHE_MLA(__nv_bfloat16);
#endif

} // namespace kernels

} // namespace tensorrt_llm
