#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <cstdint>
#include <immintrin.h>
#include <cstring>


torch::Tensor packed_ternary_linear_cpu_parallel(
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
        "OPT003 currently requires FP16 input"
    );

    TORCH_CHECK(
        scales.scalar_type() == at::kHalf,
        "OPT003 currently requires FP16 scales"
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

    const int64_t batch_size =
        input_c.size(0);

    const int64_t in_features =
        input_c.size(1);

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
        packed_c.numel() * 4
            == expected_codes,
        "packed-code size does not match matrix shape"
    );

    TORCH_CHECK(
        scales_c.numel()
            == out_features * groups_per_row,
        "scale count does not match matrix shape"
    );

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

    /*
     * OPT003 controlled change:
     *
     * OPT001:
     *     sequential batch/output-row loops
     *
     * OPT003:
     *     parallelize independent (batch, output-row)
     *     jobs using PyTorch's CPU thread pool.
     *
     * The arithmetic inside each output element remains
     * unchanged from OPT001.
     */

    const int64_t total_jobs =
        batch_size * out_features;

    static constexpr float positive_lut[4] = {
        0.0f, 0.0f, 1.0f, 0.0f
    };

    static constexpr float negative_lut[4] = {
        1.0f, 0.0f, 0.0f, 0.0f
    };

    at::parallel_for(
        0,
        total_jobs,
        64,
        [&](int64_t begin, int64_t end) {

            for (int64_t job = begin;
                 job < end;
                 ++job) {

                const int64_t b =
                    job / out_features;

                const int64_t out =
                    job % out_features;

                const at::Half* input_row =
                    input_ptr
                    + b * in_features;

                const uint8_t* packed_row =
                    packed_ptr
                    + out * bytes_per_row;

                float output_acc = 0.0f;

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

                    /*
                     * OPT008 controlled change:
                     *
                     * Decode eight packed bytes at once.
                     *
                     * Eight bytes contain 32 ternary codes.
                     * AVX2 expands these into 32 uint8 code
                     * values.
                     *
                     * IMPORTANT:
                     * Activation conversion and FP32
                     * accumulation remain scalar and occur
                     * in the original logical input order.
                     *
                     * This avoids changing reduction order
                     * in the first SIMD experiment.
                     */

                    alignas(32) uint8_t decoded_codes[32];

                    int64_t byte_index = 0;

                    for (;
                         byte_index + 8 <= group_bytes;
                         byte_index += 8) {

                        uint64_t packed8 = 0;

                        std::memcpy(
                            &packed8,
                            packed_row
                                + group_byte_start
                                + byte_index,
                            sizeof(uint64_t)
                        );

                        /*
                         * Put the eight packed bytes into the
                         * low 64 bits.
                         */
                        const __m128i packed128 =
                            _mm_cvtsi64_si128(
                                static_cast<long long>(
                                    packed8
                                )
                            );

                        /*
                         * Expand 8 uint8 values to 8 uint32
                         * lanes.
                         */
                        const __m256i packed32 =
                            _mm256_cvtepu8_epi32(
                                packed128
                            );

                        const __m256i mask =
                            _mm256_set1_epi32(0x3);

                        const __m256i c0 =
                            _mm256_and_si256(
                                packed32,
                                mask
                            );

                        const __m256i c1 =
                            _mm256_and_si256(
                                _mm256_srli_epi32(
                                    packed32,
                                    2
                                ),
                                mask
                            );

                        const __m256i c2 =
                            _mm256_and_si256(
                                _mm256_srli_epi32(
                                    packed32,
                                    4
                                ),
                                mask
                            );

                        const __m256i c3 =
                            _mm256_and_si256(
                                _mm256_srli_epi32(
                                    packed32,
                                    6
                                ),
                                mask
                            );

                        alignas(32) uint32_t t0[8];
                        alignas(32) uint32_t t1[8];
                        alignas(32) uint32_t t2[8];
                        alignas(32) uint32_t t3[8];

                        _mm256_store_si256(
                            reinterpret_cast<__m256i*>(t0),
                            c0
                        );

                        _mm256_store_si256(
                            reinterpret_cast<__m256i*>(t1),
                            c1
                        );

                        _mm256_store_si256(
                            reinterpret_cast<__m256i*>(t2),
                            c2
                        );

                        _mm256_store_si256(
                            reinterpret_cast<__m256i*>(t3),
                            c3
                        );

                        for (int lane = 0;
                             lane < 8;
                             ++lane) {

                            const uint8_t code0 =
                                static_cast<uint8_t>(
                                    t0[lane]
                                );

                            const uint8_t code1 =
                                static_cast<uint8_t>(
                                    t1[lane]
                                );

                            const uint8_t code2 =
                                static_cast<uint8_t>(
                                    t2[lane]
                                );

                            const uint8_t code3 =
                                static_cast<uint8_t>(
                                    t3[lane]
                                );

                            TORCH_CHECK(
                                code0 != 0x3
                                && code1 != 0x3
                                && code2 != 0x3
                                && code3 != 0x3,
                                "reserved ternary code "
                                "11 encountered"
                            );

                            const int64_t input_base =
                                group_input_start
                                + (
                                    byte_index
                                    + lane
                                ) * 4;

                            const float x0 =
                                static_cast<float>(
                                    input_row[
                                        input_base
                                    ]
                                );

                            const float x1 =
                                static_cast<float>(
                                    input_row[
                                        input_base + 1
                                    ]
                                );

                            const float x2 =
                                static_cast<float>(
                                    input_row[
                                        input_base + 2
                                    ]
                                );

                            const float x3 =
                                static_cast<float>(
                                    input_row[
                                        input_base + 3
                                    ]
                                );

                            positive_sum +=
                                x0
                                * positive_lut[
                                    code0
                                ];

                            negative_sum +=
                                x0
                                * negative_lut[
                                    code0
                                ];

                            positive_sum +=
                                x1
                                * positive_lut[
                                    code1
                                ];

                            negative_sum +=
                                x1
                                * negative_lut[
                                    code1
                                ];

                            positive_sum +=
                                x2
                                * positive_lut[
                                    code2
                                ];

                            negative_sum +=
                                x2
                                * negative_lut[
                                    code2
                                ];

                            positive_sum +=
                                x3
                                * positive_lut[
                                    code3
                                ];

                            negative_sum +=
                                x3
                                * negative_lut[
                                    code3
                                ];
                        }
                    }

                    /*
                     * Scalar tail.
                     *
                     * G128 currently gives exactly 32 bytes
                     * per group, so the frozen experiment
                     * should normally have no tail.
                     *
                     * Keeping this path makes the operator
                     * structurally safe for compatible group
                     * sizes not divisible by eight.
                     */
                    for (;
                         byte_index < group_bytes;
                         ++byte_index) {

                        const uint8_t packed =
                            packed_row[
                                group_byte_start
                                + byte_index
                            ];

                        const uint8_t code0 =
                            packed & 0x3;

                        const uint8_t code1 =
                            (packed >> 2) & 0x3;

                        const uint8_t code2 =
                            (packed >> 4) & 0x3;

                        const uint8_t code3 =
                            (packed >> 6) & 0x3;

                        TORCH_CHECK(
                            code0 != 0x3
                            && code1 != 0x3
                            && code2 != 0x3
                            && code3 != 0x3,
                            "reserved ternary code "
                            "11 encountered"
                        );

                        const int64_t input_base =
                            group_input_start
                            + byte_index * 4;

                        const float x0 =
                            static_cast<float>(
                                input_row[
                                    input_base
                                ]
                            );

                        const float x1 =
                            static_cast<float>(
                                input_row[
                                    input_base + 1
                                ]
                            );

                        const float x2 =
                            static_cast<float>(
                                input_row[
                                    input_base + 2
                                ]
                            );

                        const float x3 =
                            static_cast<float>(
                                input_row[
                                    input_base + 3
                                ]
                            );

                        positive_sum +=
                            x0 * positive_lut[code0];

                        negative_sum +=
                            x0 * negative_lut[code0];

                        positive_sum +=
                            x1 * positive_lut[code1];

                        negative_sum +=
                            x1 * negative_lut[code1];

                        positive_sum +=
                            x2 * positive_lut[code2];

                        negative_sum +=
                            x2 * negative_lut[code2];

                        positive_sum +=
                            x3 * positive_lut[code3];

                        negative_sum +=
                            x3 * negative_lut[code3];
                    }

                    const int64_t scale_index =
                        out
                        * groups_per_row
                        + group;

                    const float scale =
                        static_cast<float>(
                            scale_ptr[
                                scale_index
                            ]
                        );

                    const float group_contribution =
                        (
                            positive_sum
                            - negative_sum
                        )
                        * scale;

                    output_acc += (
                        group_contribution
                    );
                }

                output_ptr[
                    b * out_features
                    + out
                ] = output_acc;
            }
        }
    );

    return output_fp32.to(
        input.scalar_type()
    );
}


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
) {
    m.def(
        "packed_ternary_linear_cpu_parallel",
        &packed_ternary_linear_cpu_parallel,
        "OPT003 parallel packed ternary linear CPU"
    );
}
