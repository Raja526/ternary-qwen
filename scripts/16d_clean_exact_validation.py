import gc
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# EXPERIMENT 016D
# Clean exact-answer validation
#
# Purpose:
#   Re-run ONLY the 10 objectively scorable tasks from Experiment 016.
#
# Configurations:
#   1. FP16
#   2. BLOCK_GUIDED_7_67
#   3. ACT_GUIDED_7_73
#
# Important:
#   - Same model
#   - Same prompts
#   - Same deterministic decoding
#   - Thinking disabled through the chat template when supported
#   - Very small generation budget
#   - Exact scoring only
# =============================================================================


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

DEVICE = "cuda"

DTYPE = torch.float16

GROUP_SIZE = 128

THRESHOLD_MULTIPLIER = 0.5

MAX_NEW_TOKENS = 16

OUTPUT_FILE = Path(
    "results/016d_clean_exact_validation.txt"
)


# =============================================================================
# EXACT VALIDATION SET
# =============================================================================

TESTS = [

    # -------------------------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------------------------

    {
        "id": "A1",
        "category": "arithmetic",
        "prompt": (
            "What is 24 multiplied by 17? "
            "Answer with only the final number."
        ),
        "expected": "408",
        "type": "numeric",
    },

    {
        "id": "A2",
        "category": "arithmetic",
        "prompt": (
            "What is 432 divided by 18? "
            "Answer with only the final number."
        ),
        "expected": "24",
        "type": "numeric",
    },

    {
        "id": "A3",
        "category": "arithmetic",
        "prompt": (
            "What is 37 plus 58? "
            "Answer with only the final number."
        ),
        "expected": "95",
        "type": "numeric",
    },

    {
        "id": "A4",
        "category": "arithmetic",
        "prompt": (
            "What is 91 minus 46? "
            "Answer with only the final number."
        ),
        "expected": "45",
        "type": "numeric",
    },


    # -------------------------------------------------------------------------
    # Word problems
    # -------------------------------------------------------------------------

    {
        "id": "W1",
        "category": "word_problem",
        "prompt": (
            "A bus travels at 55 kilometers per hour "
            "for 4 hours. How many kilometers does it travel? "
            "Answer with only the final number."
        ),
        "expected": "220",
        "type": "numeric",
    },

    {
        "id": "W2",
        "category": "word_problem",
        "prompt": (
            "A runner covers 8 kilometers per hour for 3 hours. "
            "How many kilometers does the runner cover in total? "
            "Answer with only the final number."
        ),
        "expected": "24",
        "type": "numeric",
    },

    {
        "id": "W3",
        "category": "word_problem",
        "prompt": (
            "A box contains 12 rows of 9 apples. "
            "How many apples are in the box? "
            "Answer with only the final number."
        ),
        "expected": "108",
        "type": "numeric",
    },


    # -------------------------------------------------------------------------
    # Factual
    # -------------------------------------------------------------------------

    {
        "id": "F1",
        "category": "factual",
        "prompt": (
            "What planet is known as the Red Planet? "
            "Answer with only one word."
        ),
        "expected": "mars",
        "type": "word",
    },

    {
        "id": "F2",
        "category": "factual",
        "prompt": (
            "What gas do humans need to breathe to survive? "
            "Answer with only one word."
        ),
        "expected": "oxygen",
        "type": "word",
    },

    {
        "id": "F3",
        "category": "factual",
        "prompt": (
            "What is the process by which plants make food "
            "using sunlight called? "
            "Answer with only one word."
        ),
        "expected": "photosynthesis",
        "type": "word",
    },
]


# =============================================================================
# CONFIGURATION MATRICES
#
# These reproduce the exact configurations already used in Exp016.
# =============================================================================


BLOCK_GUIDED = [

    # MLP
    "model.layers.6.mlp.gate_proj",
    "model.layers.6.mlp.up_proj",
    "model.layers.6.mlp.down_proj",

    "model.layers.12.mlp.gate_proj",
    "model.layers.12.mlp.up_proj",
    "model.layers.12.mlp.down_proj",


    # Linear attention
    "model.layers.4.linear_attn.in_proj_qkv",
    "model.layers.4.linear_attn.in_proj_z",
    "model.layers.4.linear_attn.out_proj",

    "model.layers.8.linear_attn.in_proj_qkv",
    "model.layers.8.linear_attn.in_proj_z",
    "model.layers.8.linear_attn.out_proj",


    # Self attention
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.k_proj",
    "model.layers.3.self_attn.v_proj",
    "model.layers.3.self_attn.o_proj",

    "model.layers.7.self_attn.q_proj",
    "model.layers.7.self_attn.k_proj",
    "model.layers.7.self_attn.v_proj",
    "model.layers.7.self_attn.o_proj",
]


ACT_GUIDED = [

    "model.layers.7.self_attn.q_proj",
    "model.layers.3.self_attn.q_proj",

    "model.layers.18.mlp.gate_proj",

    "model.layers.12.linear_attn.in_proj_z",

    "model.layers.11.self_attn.q_proj",

    "model.layers.6.mlp.gate_proj",

    "model.layers.8.linear_attn.in_proj_z",

    "model.layers.12.linear_attn.in_proj_qkv",

    "model.layers.4.linear_attn.in_proj_z",

    "model.layers.12.mlp.gate_proj",

    "model.layers.4.linear_attn.in_proj_qkv",

    "model.layers.8.linear_attn.in_proj_qkv",

    "model.layers.7.self_attn.k_proj",

    "model.layers.3.self_attn.v_proj",

    "model.layers.11.self_attn.k_proj",

    "model.layers.18.mlp.up_proj",

    "model.layers.3.self_attn.k_proj",

    "model.layers.12.mlp.up_proj",
]


CONFIGS = {
    "FP16": [],
    "BLOCK_GUIDED_7_67": BLOCK_GUIDED,
    "ACT_GUIDED_7_73": ACT_GUIDED,
}


# =============================================================================
# OUTPUT
# =============================================================================

output_lines = []


def emit(line=""):
    print(line)
    output_lines.append(str(line))


# =============================================================================
# MODEL HELPERS
# =============================================================================

def load_model():

    emit()
    emit("Loading fresh FP16 model...")


    start = time.perf_counter()


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        device_map=DEVICE,
    )


    model.eval()


    elapsed = (
        time.perf_counter()
        - start
    )


    emit(
        f"Model loaded in "
        f"{elapsed:.2f} sec"
    )


    return model


def unload_model(model):

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# MODULE LOOKUP
# =============================================================================

def get_module_by_name(
    model,
    module_name,
):

    modules = dict(
        model.named_modules()
    )


    if module_name not in modules:

        raise KeyError(
            f"Module not found: "
            f"{module_name}"
        )


    module = modules[
        module_name
    ]


    if not isinstance(
        module,
        torch.nn.Linear,
    ):

        raise TypeError(
            f"{module_name} is not "
            f"torch.nn.Linear"
        )


    return module


# =============================================================================
# OPTIMIZED-SCALE GROUP-WISE TERNARIZATION
#
# Frozen method from Exp011 onward:
#
# initial_alpha = mean(abs(group))
# threshold     = 0.5 * initial_alpha
#
# codes:
#   +1 if w > threshold
#    0 if |w| <= threshold
#   -1 if w < -threshold
#
# After codes are fixed:
#
# alpha_opt =
#     mean(abs(nonzero selected weights))
#
# reconstructed:
#
# W_ternary = alpha_opt * code
# =============================================================================

@torch.no_grad()
def ternarize_weight(
    weight,
):

    original_shape = (
        weight.shape
    )


    flat = (
        weight
        .detach()
        .float()
        .reshape(-1)
    )


    n = flat.numel()


    reconstructed = (
        torch.empty_like(
            flat
        )
    )


    total_error_sq = 0.0

    total_weight_sq = 0.0

    scale_ratios = []


    for start in range(
        0,
        n,
        GROUP_SIZE,
    ):

        end = min(
            start + GROUP_SIZE,
            n,
        )


        group = flat[
            start:end
        ]


        abs_group = (
            group.abs()
        )


        initial_alpha = (
            abs_group.mean()
        )


        threshold = (
            THRESHOLD_MULTIPLIER
            * initial_alpha
        )


        codes = torch.zeros_like(
            group
        )


        codes[
            group > threshold
        ] = 1.0


        codes[
            group < -threshold
        ] = -1.0


        nonzero_mask = (
            codes != 0
        )


        if nonzero_mask.any():

            alpha_opt = (
                abs_group[
                    nonzero_mask
                ].mean()
            )

        else:

            alpha_opt = (
                initial_alpha
            )


        ternary_group = (
            codes
            * alpha_opt
        )


        reconstructed[
            start:end
        ] = ternary_group


        diff = (
            group
            - ternary_group
        )


        total_error_sq += (
            diff.pow(2)
            .sum()
            .item()
        )


        total_weight_sq += (
            group.pow(2)
            .sum()
            .item()
        )


        if initial_alpha.item() > 0:

            scale_ratios.append(
                (
                    alpha_opt
                    / initial_alpha
                ).item()
            )


    relative_error = (
        (
            total_error_sq
            / total_weight_sq
        ) ** 0.5
        if total_weight_sq > 0
        else 0.0
    )


    mean_scale_ratio = (
        sum(scale_ratios)
        / len(scale_ratios)
        if scale_ratios
        else 0.0
    )


    reconstructed = (
        reconstructed
        .reshape(
            original_shape
        )
        .to(
            dtype=weight.dtype,
            device=weight.device,
        )
    )


    return (
        reconstructed,
        relative_error,
        mean_scale_ratio,
    )


# =============================================================================
# APPLY CONFIGURATION
# =============================================================================

@torch.no_grad()
def apply_configuration(
    model,
    module_names,
):

    if not module_names:

        return {
            "matrices": 0,
            "parameters": 0,
            "mean_error": 0.0,
            "max_error": 0.0,
            "mean_scale_ratio": 0.0,
        }


    errors = []

    scale_ratios = []

    total_parameters = 0


    for module_name in module_names:

        module = get_module_by_name(
            model,
            module_name,
        )


        parameter_count = (
            module.weight.numel()
        )


        (
            ternary_weight,
            relative_error,
            scale_ratio,
        ) = ternarize_weight(
            module.weight
        )


        module.weight.copy_(
            ternary_weight
        )


        total_parameters += (
            parameter_count
        )


        errors.append(
            relative_error
        )


        scale_ratios.append(
            scale_ratio
        )


    total_model_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


    coverage = (
        total_parameters
        / total_model_parameters
        * 100
    )


    return {
        "matrices":
            len(module_names),

        "parameters":
            total_parameters,

        "coverage":
            coverage,

        "mean_error":
            sum(errors)
            / len(errors),

        "max_error":
            max(errors),

        "mean_scale_ratio":
            sum(scale_ratios)
            / len(scale_ratios),
    }


# =============================================================================
# CHAT TEMPLATE
# =============================================================================

def build_inputs(
    tokenizer,
    prompt,
):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]


    # -------------------------------------------------------------------------
    # First try Qwen's explicit non-thinking mode.
    # -------------------------------------------------------------------------

    try:

        text = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )


        thinking_mode = (
            "enable_thinking=False"
        )


    except TypeError:

        # ---------------------------------------------------------------------
        # Compatibility fallback.
        # ---------------------------------------------------------------------

        text = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


        thinking_mode = (
            "chat-template-default"
        )


    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )


    inputs = {
        key: value.to(
            DEVICE
        )
        for key, value
        in inputs.items()
    }


    return (
        inputs,
        thinking_mode,
    )


# =============================================================================
# GENERATE
# =============================================================================

@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt,
):

    (
        inputs,
        thinking_mode,
    ) = build_inputs(
        tokenizer,
        prompt,
    )


    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )


    if torch.cuda.is_available():
        torch.cuda.synchronize()


    start = time.perf_counter()


    output = model.generate(
        **inputs,

        max_new_tokens=
            MAX_NEW_TOKENS,

        do_sample=False,

        use_cache=True,

        pad_token_id=
            tokenizer.eos_token_id,

        eos_token_id=
            tokenizer.eos_token_id,
    )


    if torch.cuda.is_available():
        torch.cuda.synchronize()


    elapsed = (
        time.perf_counter()
        - start
    )


    generated_ids = (
        output[
            0,
            input_length:
        ]
    )


    generated_tokens = (
        generated_ids.numel()
    )


    decoded = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


    tokens_per_second = (
        generated_tokens
        / elapsed
        if elapsed > 0
        else 0.0
    )


    return {
        "text":
            decoded,

        "tokens":
            generated_tokens,

        "time":
            elapsed,

        "tokens_per_second":
            tokens_per_second,

        "thinking_mode":
            thinking_mode,
    }


# =============================================================================
# STRICT NORMALIZATION
# =============================================================================

def normalize_answer(
    text,
    answer_type,
):

    cleaned = (
        text
        .strip()
        .lower()
    )


    # Remove markdown bold markers.
    cleaned = cleaned.replace(
        "**",
        "",
    )


    # Remove surrounding whitespace again.
    cleaned = cleaned.strip()


    if answer_type == "numeric":

        # ---------------------------------------------------------------------
        # Strict numeric response.
        #
        # Accepted:
        #   408
        #   408.
        #   220 km
        #   220 kilometers
        #
        # Rejected:
        #   Answer: 408
        #   24 * 17 = 408
        #   long reasoning
        #
        # The prompt explicitly asks for only the final number.
        # ---------------------------------------------------------------------

        match = re.fullmatch(
            r"(-?\d+(?:\.\d+)?)"
            r"\s*"
            r"(?:"
            r"km|"
            r"kilometers?|"
            r"apples?"
            r")?"
            r"[.!]?",
            cleaned,
            flags=re.IGNORECASE,
        )


        if match:

            return (
                match.group(1)
            )


        return None


    if answer_type == "word":

        # ---------------------------------------------------------------------
        # Strict one-word response.
        # ---------------------------------------------------------------------

        match = re.fullmatch(
            r"([a-z]+)"
            r"[.!]?",
            cleaned,
            flags=re.IGNORECASE,
        )


        if match:

            return (
                match.group(1)
                .lower()
            )


        return None


    raise ValueError(
        f"Unknown answer type: "
        f"{answer_type}"
    )


# =============================================================================
# SCORE
# =============================================================================

def score_answer(
    generated_text,
    expected,
    answer_type,
):

    normalized = normalize_answer(
        generated_text,
        answer_type,
    )


    expected_normalized = (
        expected
        .strip()
        .lower()
    )


    if normalized is None:

        return {
            "normalized": None,
            "status": "FORMAT_FAIL",
        }


    if normalized == expected_normalized:

        return {
            "normalized":
                normalized,

            "status":
                "PASS",
        }


    return {
        "normalized":
            normalized,

        "status":
            "WRONG",
    }


# =============================================================================
# LOAD TOKENIZER ONCE
# =============================================================================

emit(
    "=" * 130
)

emit(
    "EXPERIMENT 016D — "
    "CLEAN EXACT-ANSWER VALIDATION"
)

emit(
    "=" * 130
)


emit(
    f"\nModel              : "
    f"{MODEL_NAME}"
)

emit(
    f"Device             : "
    f"{DEVICE}"
)

emit(
    f"Dtype              : "
    f"{DTYPE}"
)

emit(
    f"Group size         : "
    f"{GROUP_SIZE}"
)

emit(
    f"Threshold          : "
    f"{THRESHOLD_MULTIPLIER} * "
    f"initial_alpha"
)

emit(
    f"Max new tokens     : "
    f"{MAX_NEW_TOKENS}"
)

emit(
    f"Exact prompts      : "
    f"{len(TESTS)}"
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)


# =============================================================================
# RUN CONFIGURATIONS
# =============================================================================

summaries = {}


for config_name, modules in (
    CONFIGS.items()
):

    emit()
    emit()
    emit(
        "#" * 130
    )

    emit(
        f"VALIDATION CASE — "
        f"{config_name}"
    )

    emit(
        "#" * 130
    )


    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Fresh FP16 model for every configuration.
    #
    # This prevents ternary changes from one case leaking into another.
    # -------------------------------------------------------------------------

    model = load_model()


    stats = apply_configuration(
        model,
        modules,
    )


    emit()
    emit(
        "Configuration validation:"
    )


    emit(
        f"Matrices           : "
        f"{stats['matrices']}"
    )


    emit(
        f"Parameters         : "
        f"{stats['parameters']:,}"
    )


    if modules:

        emit(
            f"Coverage           : "
            f"{stats['coverage']:.4f}%"
        )

        emit(
            f"Mean rel. error    : "
            f"{stats['mean_error']:.6f}"
        )

        emit(
            f"Max rel. error     : "
            f"{stats['max_error']:.6f}"
        )

        emit(
            f"Mean scale ratio   : "
            f"{stats['mean_scale_ratio']:.6f}"
        )


    counts = {
        "PASS": 0,
        "WRONG": 0,
        "FORMAT_FAIL": 0,
    }


    category_counts = {}


    total_generated_tokens = 0

    total_generation_time = 0.0

    thinking_modes = set()


    for test in TESTS:

        emit()
        emit(
            "-" * 130
        )

        emit(
            f"[{test['id']}] "
            f"{test['category']}"
        )

        emit(
            f"Prompt   : "
            f"{test['prompt']}"
        )

        emit(
            f"Expected : "
            f"{test['expected']}"
        )


        result = generate_answer(
            model,
            tokenizer,
            test["prompt"],
        )


        score = score_answer(
            result["text"],
            test["expected"],
            test["type"],
        )


        thinking_modes.add(
            result[
                "thinking_mode"
            ]
        )


        status = (
            score["status"]
        )


        counts[
            status
        ] += 1


        category = (
            test[
                "category"
            ]
        )


        if category not in (
            category_counts
        ):

            category_counts[
                category
            ] = {
                "PASS": 0,
                "WRONG": 0,
                "FORMAT_FAIL": 0,
                "TOTAL": 0,
            }


        category_counts[
            category
        ][
            status
        ] += 1


        category_counts[
            category
        ][
            "TOTAL"
        ] += 1


        total_generated_tokens += (
            result["tokens"]
        )


        total_generation_time += (
            result["time"]
        )


        emit(
            f"Output   : "
            f"{repr(result['text'])}"
        )

        emit(
            f"Parsed   : "
            f"{score['normalized']}"
        )

        emit(
            f"Status   : "
            f"{status}"
        )

        emit(
            f"Tokens   : "
            f"{result['tokens']}"
        )

        emit(
            f"Time     : "
            f"{result['time']:.4f} sec"
        )

        emit(
            f"Tok/s    : "
            f"{result['tokens_per_second']:.2f}"
        )

        emit(
            f"Template : "
            f"{result['thinking_mode']}"
        )


    total = len(
        TESTS
    )


    accuracy = (
        counts["PASS"]
        / total
        * 100
    )


    aggregate_tps = (
        total_generated_tokens
        / total_generation_time
        if total_generation_time > 0
        else 0.0
    )


    emit()
    emit(
        "=" * 130
    )

    emit(
        f"SUMMARY — "
        f"{config_name}"
    )

    emit(
        "=" * 130
    )


    emit(
        f"PASS              : "
        f"{counts['PASS']}"
    )

    emit(
        f"WRONG             : "
        f"{counts['WRONG']}"
    )

    emit(
        f"FORMAT_FAIL       : "
        f"{counts['FORMAT_FAIL']}"
    )

    emit(
        f"TOTAL             : "
        f"{total}"
    )

    emit(
        f"Strict accuracy   : "
        f"{accuracy:.2f}%"
    )


    emit()

    emit(
        "Category breakdown:"
    )


    for category in [
        "arithmetic",
        "word_problem",
        "factual",
    ]:

        values = (
            category_counts[
                category
            ]
        )


        emit(
            f"{category:<14} "
            f"PASS={values['PASS']} "
            f"WRONG={values['WRONG']} "
            f"FORMAT_FAIL="
            f"{values['FORMAT_FAIL']} "
            f"TOTAL={values['TOTAL']}"
        )


    emit()

    emit(
        f"Generated tokens  : "
        f"{total_generated_tokens}"
    )

    emit(
        f"Generation time   : "
        f"{total_generation_time:.4f} sec"
    )

    emit(
        f"Aggregate tok/s   : "
        f"{aggregate_tps:.2f}"
    )

    emit(
        f"Template mode(s)  : "
        f"{sorted(thinking_modes)}"
    )


    summaries[
        config_name
    ] = {
        "PASS":
            counts["PASS"],

        "WRONG":
            counts["WRONG"],

        "FORMAT_FAIL":
            counts[
                "FORMAT_FAIL"
            ],

        "accuracy":
            accuracy,

        "tokens":
            total_generated_tokens,

        "time":
            total_generation_time,

        "tps":
            aggregate_tps,
    }


    unload_model(
        model
    )


# =============================================================================
# FINAL COMPARISON
# =============================================================================

emit()
emit()
emit(
    "=" * 130
)

emit(
    "FINAL CLEAN EXACT-ANSWER COMPARISON"
)

emit(
    "=" * 130
)


emit(
    f"{'Configuration':<24}"
    f"{'PASS':>8}"
    f"{'WRONG':>10}"
    f"{'FORMAT':>10}"
    f"{'Accuracy':>14}"
    f"{'Tok/s':>12}"
)


emit(
    "-" * 130
)


for config_name in CONFIGS:

    summary = (
        summaries[
            config_name
        ]
    )


    emit(
        f"{config_name:<24}"
        f"{summary['PASS']:>8}"
        f"{summary['WRONG']:>10}"
        f"{summary['FORMAT_FAIL']:>10}"
        f"{summary['accuracy']:>13.2f}%"
        f"{summary['tps']:>12.2f}"
    )


# =============================================================================
# PASS-COUNT RETENTION
# =============================================================================

emit()
emit(
    "=" * 130
)

emit(
    "PASS-COUNT RETENTION RELATIVE TO FP16"
)

emit(
    "=" * 130
)


fp16_pass = (
    summaries[
        "FP16"
    ][
        "PASS"
    ]
)


for config_name in [
    "BLOCK_GUIDED_7_67",
    "ACT_GUIDED_7_73",
]:

    ternary_pass = (
        summaries[
            config_name
        ][
            "PASS"
        ]
    )


    if fp16_pass > 0:

        retention = (
            ternary_pass
            / fp16_pass
            * 100
        )


        emit(
            f"{config_name:<24}: "
            f"{ternary_pass}/"
            f"{fp16_pass} "
            f"= {retention:.2f}%"
        )


    else:

        emit(
            f"{config_name:<24}: "
            "undefined"
        )


# =============================================================================
# SCIENTIFIC NOTES
# =============================================================================

emit()
emit(
    "=" * 130
)

emit(
    "SCIENTIFIC NOTES"
)

emit(
    "=" * 130
)


emit(
    "1. Each configuration was evaluated "
    "from a fresh FP16 model load."
)

emit(
    "2. The ternary method is frozen: "
    "group size 128, threshold 0.5 * "
    "initial mean-absolute scale, followed "
    "by optimized nonzero scale."
)

emit(
    "3. Thinking was requested to be "
    "disabled through the Qwen chat "
    "template when supported."
)

emit(
    "4. Generation is deterministic "
    "(do_sample=False)."
)

emit(
    "5. Exact prompts use a small "
    "generation budget to reduce "
    "post-answer degeneration."
)

emit(
    "6. PASS requires a clean response "
    "matching the expected answer."
)

emit(
    "7. WRONG means a clean answer was "
    "produced but its value was incorrect."
)

emit(
    "8. FORMAT_FAIL means the model did "
    "not follow the requested short-answer "
    "format closely enough for strict "
    "automatic scoring."
)

emit(
    "9. FORMAT_FAIL is kept separate from "
    "WRONG because instruction-following "
    "failure and factual/numerical error "
    "are different failure modes."
)

emit(
    "10. Tokens/sec must not be interpreted "
    "as packed ternary acceleration because "
    "the ternary matrices are reconstructed "
    "and stored as FP16 tensors."
)


emit()
emit(
    "=" * 130
)

emit(
    "EXPERIMENT 016D COMPLETE"
)

emit(
    "=" * 130
)


# =============================================================================
# SAVE
# =============================================================================

OUTPUT_FILE.write_text(
    "\n".join(
        output_lines
    )
    + "\n",
    encoding="utf-8",
)


print(
    f"\nSaved results to: "
    f"{OUTPUT_FILE}"
)
