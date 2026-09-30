import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load_inline


EXPECTED_CUDA_HOME = "/usr/local/cuda-12.6"
EXPECTED_ARCH = "6.1"


cpp_source = r"""
#include <torch/extension.h>

torch::Tensor add_one_cuda(torch::Tensor input);

torch::Tensor add_one(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        "input must be float32"
    );

    return add_one_cuda(input);
}
"""


cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_one_kernel(
    const float* input,
    float* output,
    int64_t n
) {
    int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < n) {
        output[i] = input[i] + 1.0f;
    }
}

torch::Tensor add_one_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int64_t n = input.numel();

    const int blocks =
        static_cast<int>((n + threads - 1) / threads);

    add_one_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}
"""


print("PyTorch             :", torch.__version__)
print("PyTorch CUDA        :", torch.version.cuda)
print("CUDA_HOME           :", CUDA_HOME)
print("TORCH_CUDA_ARCH_LIST:",
      os.environ.get("TORCH_CUDA_ARCH_LIST"))
print("CUDA available      :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU                 :", torch.cuda.get_device_name(0))
print("Capability          :", torch.cuda.get_device_capability(0))

if CUDA_HOME != EXPECTED_CUDA_HOME:
    raise RuntimeError(
        f"Expected CUDA_HOME={EXPECTED_CUDA_HOME}, got {CUDA_HOME}"
    )

if os.environ.get("TORCH_CUDA_ARCH_LIST") != EXPECTED_ARCH:
    raise RuntimeError(
        f"Expected TORCH_CUDA_ARCH_LIST={EXPECTED_ARCH}"
    )

if torch.version.cuda != "12.6":
    raise RuntimeError(
        f"Expected PyTorch CUDA 12.6, got {torch.version.cuda}"
    )

if torch.cuda.get_device_capability(0) != (6, 1):
    raise RuntimeError(
        "Expected compute capability (6, 1)"
    )


extension = load_inline(
    name="gpu002_cuda126_sm61_validation",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["add_one"],
    extra_cuda_cflags=[
        "-gencode=arch=compute_61,code=sm_61",
    ],
    verbose=True,
)


x = torch.tensor(
    [1.0, 2.0, 3.0, 4.0],
    device="cuda",
    dtype=torch.float32,
)

expected = torch.tensor(
    [2.0, 3.0, 4.0, 5.0],
    device="cuda",
    dtype=torch.float32,
)

actual = extension.add_one(x)

torch.cuda.synchronize()

print("Input               :", x.cpu().tolist())
print("Expected            :", expected.cpu().tolist())
print("Actual              :", actual.cpu().tolist())

if not torch.equal(actual, expected):
    raise RuntimeError(
        "CUDA extension produced incorrect output"
    )

print()
print("GPU002 CUDA TOOLCHAIN VALIDATION: PASS")
