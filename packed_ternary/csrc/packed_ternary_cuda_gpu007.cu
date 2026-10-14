#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>


__global__ void packed_ternary_linear_parallel_kernel(
    const half* __restrict__ input,
    const uint8_t* __restrict__ packed_codes,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    int64_t group_size
) {
    const int64_t job = static_cast<int64_t>(blockIdx.x);
    const int64_t total_jobs = batch_size * out_features;

    if (job >= total_jobs) {
        return;
    }

    const int64_t batch_row = job / out_features;
    const int64_t out_row = job % out_features;
    const int64_t groups_per_row = in_features / group_size;
    const int64_t bytes_per_row = in_features / 4;
    const int64_t group_bytes = group_size / 4;

    const half* input_row = input + batch_row * in_features;
    const uint8_t* packed_row = packed_codes + out_row * bytes_per_row;
    const int thread = threadIdx.x;

    float local_sum = 0.0f;

    /*
     * Frozen ENG001 encoding:
     *
     *   -1 -> 00
     *    0 -> 01
     *   +1 -> 10
     *   11 -> reserved
     *
     * GPU007 assumes validated packed input and does not reconstruct an
     * ordinary FP16 weight matrix.
     */
    for (int64_t byte_index = thread;
         byte_index < bytes_per_row;
         byte_index += blockDim.x) {
        const uint8_t packed = packed_row[byte_index];
        const uint8_t code0 = packed & 0x3;
        const uint8_t code1 = (packed >> 2) & 0x3;
        const uint8_t code2 = (packed >> 4) & 0x3;
        const uint8_t code3 = (packed >> 6) & 0x3;

        const int64_t group = byte_index / group_bytes;
        const int64_t input_base = byte_index * 4;
        const float scale = __half2float(
            scales[out_row * groups_per_row + group]
        );

        float contribution = 0.0f;
        if (code0 == 0x2) {
            contribution += __half2float(input_row[input_base]);
        } else if (code0 == 0x0) {
            contribution -= __half2float(input_row[input_base]);
        }
        if (code1 == 0x2) {
            contribution += __half2float(input_row[input_base + 1]);
        } else if (code1 == 0x0) {
            contribution -= __half2float(input_row[input_base + 1]);
        }
        if (code2 == 0x2) {
            contribution += __half2float(input_row[input_base + 2]);
        } else if (code2 == 0x0) {
            contribution -= __half2float(input_row[input_base + 2]);
        }
        if (code3 == 0x2) {
            contribution += __half2float(input_row[input_base + 3]);
        } else if (code3 == 0x0) {
            contribution -= __half2float(input_row[input_base + 3]);
        }

        local_sum += contribution * scale;
    }

    __shared__ float partials[256];
    partials[thread] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            partials[thread] += partials[thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        output[job] = __float2half(partials[0]);
    }
}


torch::Tensor packed_ternary_linear_gpu007(
    torch::Tensor input,
    torch::Tensor packed_codes,
    torch::Tensor scales,
    int64_t out_features,
    int64_t group_size
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(packed_codes.is_cuda(), "packed_codes must be CUDA");
    TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "GPU007 requires FP16 input");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "GPU007 requires FP16 scales");
    TORCH_CHECK(packed_codes.scalar_type() == at::kByte, "packed_codes must be uint8");
    TORCH_CHECK(input.dim() == 2, "input must be 2-D");
    TORCH_CHECK(packed_codes.dim() == 1, "packed_codes must be 1-D");
    TORCH_CHECK(group_size > 0, "group_size must be positive");

    auto input_c = input.contiguous();
    auto packed_c = packed_codes.contiguous();
    auto scales_c = scales.contiguous();
    const int64_t batch_size = input_c.size(0);
    const int64_t in_features = input_c.size(1);

    TORCH_CHECK(in_features % group_size == 0,
                "in_features must be divisible by group_size");
    TORCH_CHECK(in_features % 4 == 0,
                "in_features must be divisible by 4");

    const int64_t groups_per_row = in_features / group_size;
    TORCH_CHECK(packed_c.numel() == out_features * in_features / 4,
                "packed_codes size mismatch");
    TORCH_CHECK(scales_c.numel() == out_features * groups_per_row,
                "scales size mismatch");

    auto output = torch::empty({batch_size, out_features}, input_c.options());
    const int threads = 256;
    const int64_t total_jobs = batch_size * out_features;
    const int blocks = static_cast<int>(total_jobs);

    packed_ternary_linear_parallel_kernel<<<
        blocks, threads, 0, at::cuda::getDefaultCUDAStream()
    >>>(
        reinterpret_cast<const half*>(input_c.data_ptr<at::Half>()),
        packed_c.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_c.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        batch_size,
        in_features,
        out_features,
        group_size
    );

    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "GPU007 CUDA kernel launch failed: ",
                cudaGetErrorString(error));
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "packed_ternary_linear",
        &packed_ternary_linear_gpu007,
        "GPU007 parallel packed ternary CUDA"
    );
}
