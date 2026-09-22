import torch
import torch.nn.functional as F


GROUP_SIZE = 128
THRESHOLD_MULTIPLIER = 0.50


@torch.no_grad()
def groupwise_ternarize_weight(
    weight,
    method="optimized",
    group_size=GROUP_SIZE,
    threshold_multiplier=THRESHOLD_MULTIPLIER,
):
    """
    Canonical group-wise ternary quantizer.

    This implementation is frozen from the quantization method used
    in Experiment 011 and subsequent optimized-scale experiments.

    Ternary codes:
        -1, 0, +1

    Code-selection scale:
        initial_alpha = mean(abs(group))

    Threshold:
        threshold_multiplier * initial_alpha

    Optimized reconstruction scale:
        mean(abs(weight)) over entries assigned a non-zero ternary code.
    """

    original_shape = weight.shape

    flat = (
        weight
        .detach()
        .float()
        .reshape(-1)
    )

    num_values = flat.numel()

    padding = (
        group_size
        - (num_values % group_size)
    ) % group_size

    if padding > 0:
        padded = F.pad(
            flat,
            (0, padding),
            mode="constant",
            value=0.0,
        )
    else:
        padded = flat

    groups = padded.reshape(
        -1,
        group_size,
    )

    # ---------------------------------------------------------
    # Step 1: original scale
    # ---------------------------------------------------------

    initial_alpha = (
        groups
        .abs()
        .mean(
            dim=1,
            keepdim=True,
        )
    )

    # ---------------------------------------------------------
    # Step 2: fixed ternary code decision
    # ---------------------------------------------------------

    threshold = (
        threshold_multiplier
        * initial_alpha
    )

    codes = torch.zeros_like(groups)

    codes[
        groups > threshold
    ] = 1.0

    codes[
        groups < -threshold
    ] = -1.0

    # ---------------------------------------------------------
    # Step 3: reconstruction scale
    # ---------------------------------------------------------

    if method == "baseline":

        reconstruction_alpha = initial_alpha

    elif method == "optimized":

        nonzero_mask = (
            codes != 0
        )

        nonzero_count = (
            nonzero_mask
            .sum(
                dim=1,
                keepdim=True,
            )
        )

        nonzero_abs_sum = (
            groups
            .abs()
            .mul(nonzero_mask)
            .sum(
                dim=1,
                keepdim=True,
            )
        )

        optimized_alpha = (
            nonzero_abs_sum
            / nonzero_count.clamp(min=1)
        )

        reconstruction_alpha = torch.where(
            nonzero_count > 0,
            optimized_alpha,
            initial_alpha,
        )

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    # ---------------------------------------------------------
    # Reconstruction
    # ---------------------------------------------------------

    reconstructed = (
        codes
        * reconstruction_alpha
    )

    # ---------------------------------------------------------
    # Code statistics
    # ---------------------------------------------------------

    valid_codes = (
        codes
        .reshape(-1)[:num_values]
    )

    negative_count = (
        valid_codes < 0
    ).sum().item()

    zero_count = (
        valid_codes == 0
    ).sum().item()

    positive_count = (
        valid_codes > 0
    ).sum().item()

    # ---------------------------------------------------------
    # Scale statistics
    # ---------------------------------------------------------

    mean_initial_alpha = (
        initial_alpha
        .mean()
        .item()
    )

    mean_reconstruction_alpha = (
        reconstruction_alpha
        .mean()
        .item()
    )

    scale_ratio = (
        mean_reconstruction_alpha
        / mean_initial_alpha
        if mean_initial_alpha > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Restore original tensor shape and dtype
    # ---------------------------------------------------------

    reconstructed = (
        reconstructed
        .reshape(-1)[:num_values]
        .reshape(original_shape)
        .to(dtype=weight.dtype)
    )

    return {
        "reconstructed": reconstructed,
        "negative": negative_count,
        "zero": zero_count,
        "positive": positive_count,
        "mean_initial_alpha": mean_initial_alpha,
        "mean_reconstruction_alpha": mean_reconstruction_alpha,
        "scale_ratio": scale_ratio,
    }


@torch.no_grad()
def ternarize_modules(
    modules,
    method="optimized",
    group_size=GROUP_SIZE,
    threshold_multiplier=THRESHOLD_MULTIPLIER,
):
    """
    Apply the canonical quantizer to an explicit list of modules.

    modules must contain:
        (name, module, component, block)

    Statistics intentionally reproduce the aggregation convention
    used in Experiment 011:
        - parameter-weighted mean relative error
        - maximum matrix relative error
        - parameter-weighted mean scale ratio
    """

    matrix_stats = []

    total_negative = 0
    total_zero = 0
    total_positive = 0

    for (
        name,
        module,
        component,
        block,
    ) in modules:

        weight = module.weight

        original_fp32 = (
            weight
            .detach()
            .float()
        )

        result = groupwise_ternarize_weight(
            weight,
            method=method,
            group_size=group_size,
            threshold_multiplier=threshold_multiplier,
        )

        reconstructed = result[
            "reconstructed"
        ]

        # Important:
        # reconstructed has already been converted back to the
        # original weight dtype, matching Experiment 011.
        reconstructed_fp32 = (
            reconstructed.float()
        )

        error = (
            original_fp32
            - reconstructed_fp32
        )

        original_norm = (
            torch.linalg.vector_norm(
                original_fp32
            ).item()
        )

        error_norm = (
            torch.linalg.vector_norm(
                error
            ).item()
        )

        relative_error = (
            error_norm / original_norm
            if original_norm > 0
            else 0.0
        )

        parameters = weight.numel()

        weight.copy_(
            reconstructed
        )

        matrix_stats.append(
            {
                "name": name,
                "component": component,
                "block": block,
                "parameters": parameters,
                "relative_error": relative_error,
                "initial_alpha":
                    result["mean_initial_alpha"],
                "reconstruction_alpha":
                    result[
                        "mean_reconstruction_alpha"
                    ],
                "scale_ratio":
                    result["scale_ratio"],
            }
        )

        total_negative += result[
            "negative"
        ]

        total_zero += result[
            "zero"
        ]

        total_positive += result[
            "positive"
        ]

        del original_fp32
        del reconstructed_fp32
        del reconstructed
        del error
        del result

    if not matrix_stats:
        raise ValueError(
            "No modules were supplied for ternarization."
        )

    total_parameters = sum(
        item["parameters"]
        for item in matrix_stats
    )

    weighted_error = sum(
        item["relative_error"]
        * item["parameters"]
        for item in matrix_stats
    ) / total_parameters

    max_error = max(
        item["relative_error"]
        for item in matrix_stats
    )

    weighted_scale_ratio = sum(
        item["scale_ratio"]
        * item["parameters"]
        for item in matrix_stats
    ) / total_parameters

    return {
        "matrix_stats": matrix_stats,
        "linear_layers": len(matrix_stats),
        "parameters": total_parameters,
        "mean_error": weighted_error,
        "max_error": max_error,
        "negative": total_negative,
        "zero": total_zero,
        "positive": total_positive,
        "mean_scale_ratio":
            weighted_scale_ratio,
    }
