#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cstdint>
#include <vector>


torch::Tensor packed_ternary_linear_cpu(
    torch::Tensor input,
    torch::Tensor packed_codes,
    torch::Tensor scales,
    int64_t out_features,
    int64_t group_size
) {
    TORCH_CHECK(
        input.device().is_cpu(),
        "input must be on CPU"
    );

    TORCH_CHECK(
        packed_codes.device().is_cpu(),
        "packed_codes must be on CPU"
    );

    TORCH_CHECK(
        scales.device().is_cpu(),
        "scales must be on CPU"
    );

    TORCH_CHECK(
        input.scalar_type() == at::kHalf,
        "OPT001 currently requires FP16 input"
    );

    TORCH_CHECK(
        scales.scalar_type() == at::kHalf,
        "OPT001 currently requires FP16 scales"
    );

    TORCH_CHECK(
        packed_codes.scalar_type() == at::kByte,
        "packed_codes must be uint8"
    );

    TORCH_CHECK(
        input.dim() == 2,
        "input must be a 2-D flattened tensor"
    );

    TORCH_CHECK(
        packed_codes.dim() == 1,
        "packed_codes must be 1-D"
    );

    TORCH_CHECK(
        group_size > 0,
        "group_size must be positive"
    );

    auto input_c = input.contiguous();
    auto packed_c = packed_codes.contiguous();
    auto scales_c = scales.contiguous();

    const int64_t batch_size = input_c.size(0);
    const int64_t in_features = input_c.size(1);

    TORCH_CHECK(
        in_features % group_size == 0,
        "in_features must be divisible by group_size"
    );

    TORCH_CHECK(
        in_features % 4 == 0,
        "in_features must be divisible by 4"
    );

    const int64_t groups_per_row =
        in_features / group_size;

    const int64_t expected_codes =
        out_features * in_features;

    TORCH_CHECK(
        packed_c.numel() * 4 == expected_codes,
        "packed-code size does not match matrix shape"
    );

    TORCH_CHECK(
        scales_c.numel()
            == out_features * groups_per_row,
        "scale count does not match matrix shape"
    );

    /*
     * Accumulate in FP32 internally.
     *
     * OPT001 is a correctness-oriented low-level prototype.
     * We will measure its numerical difference from ENG011
     * explicitly rather than assuming bit-exact FP16 behavior.
     */
    auto output_fp32 = torch::zeros(
        {batch_size, out_features},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
    );

    const at::Half* input_ptr =
        input_c.data_ptr<at::Half>();

    const uint8_t* packed_ptr =
        packed_c.data_ptr<uint8_t>();

    const at::Half* scale_ptr =
        scales_c.data_ptr<at::Half>();

    float* output_ptr =
        output_fp32.data_ptr<float>();

    const int64_t bytes_per_row =
        in_features / 4;

    for (int64_t b = 0; b < batch_size; ++b) {

        const at::Half* input_row =
            input_ptr + b * in_features;

        for (int64_t out = 0;
             out < out_features;
             ++out) {

            float output_acc = 0.0f;

            const uint8_t* packed_row =
                packed_ptr + out * bytes_per_row;

            for (int64_t group = 0;
                 group < groups_per_row;
                 ++group) {

                float positive_sum = 0.0f;
                float negative_sum = 0.0f;

                const int64_t group_input_start =
                    group * group_size;

                const int64_t group_byte_start =
                    group_input_start / 4;

                const int64_t group_bytes =
                    group_size / 4;

                for (int64_t byte_index = 0;
                     byte_index < group_bytes;
                     ++byte_index) {

                    const uint8_t packed =
                        packed_row[
                            group_byte_start
                            + byte_index
                        ];

                    for (int shift_index = 0;
                         shift_index < 4;
                         ++shift_index) {

                        const uint8_t encoded =
                            (packed
                             >> (2 * shift_index))
                            & 0x3;

                        TORCH_CHECK(
                            encoded != 0x3,
                            "reserved ternary code 11 encountered"
                        );

                        const int64_t input_index =
                            group_input_start
                            + byte_index * 4
                            + shift_index;

                        const float x =
                            static_cast<float>(
                                input_row[input_index]
                            );

                        /*
                         * Frozen encoding:
                         *
                         * 00 -> -1
                         * 01 ->  0
                         * 10 -> +1
                         */
                        if (encoded == 0x2) {
                            positive_sum += x;
                        }
                        else if (encoded == 0x0) {
                            negative_sum += x;
                        }
                    }
                }

                const int64_t scale_index =
                    out * groups_per_row
                    + group;

                const float scale =
                    static_cast<float>(
                        scale_ptr[scale_index]
                    );

                const float group_contribution =
                    (positive_sum - negative_sum)
                    * scale;

                output_acc += group_contribution;
            }

            output_ptr[
                b * out_features + out
            ] = output_acc;
        }
    }

    return output_fp32.to(input.scalar_type());
}


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
) {
    m.def(
        "packed_ternary_linear_cpu",
        &packed_ternary_linear_cpu,
        "OPT001 packed ternary linear CPU"
    );
}
