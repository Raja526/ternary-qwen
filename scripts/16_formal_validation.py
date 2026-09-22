import gc
import re
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "Qwen/Qwen3.5-0.8B"

DEVICE = "cuda"
DTYPE = torch.float16

GROUP_SIZE = 128
THRESHOLD_MULTIPLIER = 0.50

MAX_NEW_TOKENS = 80

TOTAL_EXPECTED_PARAMS = 752_393_024


# =============================================================================
# BLOCK-GUIDED CONFIGURATION
# =============================================================================
#
# Exact MEDIUM configuration from Experiments 011–013.
#
# Expected:
#   20 matrices
#   57,671,680 parameters
#   7.67% total model coverage
# =============================================================================

BLOCK_GUIDED_BLOCKS = {
    "mlp": [6, 12],
    "linear_attn": [4, 8],
    "self_attn": [3, 7],
}


# =============================================================================
# ACTIVATION-GUIDED CONFIGURATION
# =============================================================================
#
# Exact first 18 matrices from Experiment 015.
#
# Expected:
#   18 matrices
#   58,195,968 parameters
#   7.73% total model coverage
# =============================================================================

ACT_GUIDED_NAMES = {
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
}


# =============================================================================
# EVALUATION SET
# =============================================================================
#
# IMPORTANT:
# These prompts are different from the calibration/evaluation prompts used
# in Experiments 003–015.
#
# Exact-answer tasks are scored automatically.
#
# Open-ended tasks are preserved in the output for later manual / benchmark
# analysis.
# =============================================================================

EVALUATION_SET = [

    # -------------------------------------------------------------------------
    # EXACT ARITHMETIC
    # -------------------------------------------------------------------------

    {
        "id": "A1",
        "category": "arithmetic",
        "prompt": (
            "Answer with only the final number. "
            "What is 24 multiplied by 17?"
        ),
        "expected": "408",
    },

    {
        "id": "A2",
        "category": "arithmetic",
        "prompt": (
            "Answer with only the final number. "
            "What is 432 divided by 18?"
        ),
        "expected": "24",
    },

    {
        "id": "A3",
        "category": "arithmetic",
        "prompt": (
            "Answer with only the final number. "
            "What is 37 plus 58?"
        ),
        "expected": "95",
    },

    {
        "id": "A4",
        "category": "arithmetic",
        "prompt": (
            "Answer with only the final number. "
            "What is 91 minus 46?"
        ),
        "expected": "45",
    },


    # -------------------------------------------------------------------------
    # EXACT WORD PROBLEMS
    # -------------------------------------------------------------------------

    {
        "id": "W1",
        "category": "word_problem",
        "prompt": (
            "Answer with only the final number. "
            "A bus travels 55 kilometers per hour for 4 hours. "
            "How many kilometers does it travel?"
        ),
        "expected": "220",
    },

    {
        "id": "W2",
        "category": "word_problem",
        "prompt": (
            "Answer with only the final number. "
            "A runner covers 8 kilometers per hour for 3 hours. "
            "How many kilometers does the runner cover?"
        ),
        "expected": "24",
    },

    {
        "id": "W3",
        "category": "word_problem",
        "prompt": (
            "Answer with only the final number. "
            "A box contains 12 rows of 9 apples. "
            "How many apples are there?"
        ),
        "expected": "108",
    },


    # -------------------------------------------------------------------------
    # EXACT FACTUAL / BASIC KNOWLEDGE
    # -------------------------------------------------------------------------

    {
        "id": "F1",
        "category": "factual",
        "prompt": (
            "Answer with only one word. "
            "What planet is known as the Red Planet?"
        ),
        "expected": "mars",
    },

    {
        "id": "F2",
        "category": "factual",
        "prompt": (
            "Answer with only one word. "
            "What gas do humans need to breathe to survive?"
        ),
        "expected": "oxygen",
    },

    {
        "id": "F3",
        "category": "factual",
        "prompt": (
            "Answer with only one word. "
            "What is the process by which plants make food using sunlight?"
        ),
        "expected": "photosynthesis",
    },


    # -------------------------------------------------------------------------
    # OPEN-ENDED SCIENCE
    # -------------------------------------------------------------------------

    {
        "id": "S1",
        "category": "science",
        "prompt": (
            "Explain in simple terms why objects fall toward the ground."
        ),
        "expected": None,
    },

    {
        "id": "S2",
        "category": "science",
        "prompt": (
            "Explain in simple terms why ice floats on liquid water."
        ),
        "expected": None,
    },

    {
        "id": "S3",
        "category": "science",
        "prompt": (
            "Explain in simple terms how rain forms."
        ),
        "expected": None,
    },

    {
        "id": "S4",
        "category": "science",
        "prompt": (
            "Explain in simple terms why the Moon appears to change shape "
            "during the month."
        ),
        "expected": None,
    },


    # -------------------------------------------------------------------------
    # OPEN-ENDED GENERAL EXPLANATION
    # -------------------------------------------------------------------------

    {
        "id": "G1",
        "category": "general",
        "prompt": (
            "Explain in simple terms why regular sleep is important "
            "for the human body."
        ),
        "expected": None,
    },

    {
        "id": "G2",
        "category": "general",
        "prompt": (
            "Explain in simple terms why washing your hands helps "
            "prevent illness."
        ),
        "expected": None,
    },

    {
        "id": "G3",
        "category": "general",
        "prompt": (
            "Explain in simple terms why saving money can be useful."
        ),
        "expected": None,
    },


    # -------------------------------------------------------------------------
    # SIMPLE INSTRUCTION FOLLOWING
    # -------------------------------------------------------------------------

    {
        "id": "I1",
        "category": "instruction",
        "prompt": (
            "Write exactly three short benefits of reading books. "
            "Use a numbered list."
        ),
        "expected": None,
    },

    {
        "id": "I2",
        "category": "instruction",
        "prompt": (
            "Give exactly three examples of renewable energy sources. "
            "Use a numbered list."
        ),
        "expected": None,
    },

    {
        "id": "I3",
        "category": "instruction",
        "prompt": (
            "In two short sentences, explain why exercise is useful."
        ),
        "expected": None,
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def print_gpu_memory(label):

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 2)
    )

    free_bytes, total_bytes = (
        torch.cuda.mem_get_info()
    )

    print(f"\nGPU MEMORY — {label}")

    print(
        f"Allocated : {allocated:.2f} MB"
    )

    print(
        f"Reserved  : {reserved:.2f} MB"
    )

    print(
        f"Free      : "
        f"{free_bytes / (1024 ** 2):.2f} MB"
    )

    print(
        f"Total     : "
        f"{total_bytes / (1024 ** 2):.2f} MB"
    )


def load_model():

    gc.collect()
    torch.cuda.empty_cache()

    print_gpu_memory(
        "before model load"
    )

    print(
        "\nLoading fresh FP16 model..."
    )

    start = time.perf_counter()

    model = (
        Qwen3_5ForCausalLM
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
    )

    model.eval()

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Model loaded in "
        f"{elapsed:.2f} sec"
    )

    print_gpu_memory(
        "after model load"
    )

    return model


def release_model(model):

    print(
        "\nReleasing model..."
    )

    model.to("cpu")

    del model

    gc.collect()
    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    print_gpu_memory(
        "after model cleanup"
    )


def get_block_index(name):

    parts = name.split(".")

    try:

        position = parts.index(
            "layers"
        )

        return int(
            parts[position + 1]
        )

    except (
        ValueError,
        IndexError,
    ):

        return None


def get_component(
    name,
    module,
):

    if not isinstance(
        module,
        torch.nn.Linear,
    ):
        return None

    lower = name.lower()

    if ".mlp." in lower:
        return "mlp"

    if ".linear_attn." in lower:
        return "linear_attn"

    if ".self_attn." in lower:
        return "self_attn"

    return None


# =============================================================================
# QUANTIZER — UNCHANGED FROM EXPERIMENTS 011–015
# =============================================================================

@torch.no_grad()
def optimized_ternary_weight(
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

    num_values = (
        flat.numel()
    )

    padding = (
        GROUP_SIZE
        - (
            num_values
            % GROUP_SIZE
        )
    ) % GROUP_SIZE


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
        GROUP_SIZE,
    )


    initial_alpha = (
        groups
        .abs()
        .mean(
            dim=1,
            keepdim=True,
        )
    )


    threshold = (
        THRESHOLD_MULTIPLIER
        * initial_alpha
    )


    codes = torch.zeros_like(
        groups
    )

    codes[
        groups > threshold
    ] = 1.0

    codes[
        groups < -threshold
    ] = -1.0


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
        .mul(
            nonzero_mask
        )
        .sum(
            dim=1,
            keepdim=True,
        )
    )


    optimized_alpha = (
        nonzero_abs_sum
        / nonzero_count.clamp(
            min=1
        )
    )


    reconstruction_alpha = (
        torch.where(
            nonzero_count > 0,
            optimized_alpha,
            initial_alpha,
        )
    )


    reconstructed = (
        codes
        * reconstruction_alpha
    )


    valid_codes = (
        codes
        .reshape(-1)[
            :num_values
        ]
    )


    negative = (
        valid_codes < 0
    ).sum().item()

    zero = (
        valid_codes == 0
    ).sum().item()

    positive = (
        valid_codes > 0
    ).sum().item()


    initial_mean = (
        initial_alpha
        .mean()
        .item()
    )


    optimized_mean = (
        reconstruction_alpha
        .mean()
        .item()
    )


    scale_ratio = (
        optimized_mean
        / initial_mean
        if initial_mean > 0
        else 0.0
    )


    reconstructed = (
        reconstructed
        .reshape(-1)[
            :num_values
        ]
        .reshape(
            original_shape
        )
        .to(
            dtype=weight.dtype
        )
    )


    return {
        "weight":
            reconstructed,

        "negative":
            negative,

        "zero":
            zero,

        "positive":
            positive,

        "scale_ratio":
            scale_ratio,
    }


# =============================================================================
# CONFIGURATION SELECTION
# =============================================================================

def is_block_guided_target(
    name,
    module,
):

    component = get_component(
        name,
        module,
    )

    if component is None:
        return False

    block = get_block_index(
        name
    )

    if block is None:
        return False

    if (
        module.weight.numel()
        < 100_000
    ):
        return False

    return (
        block
        in BLOCK_GUIDED_BLOCKS.get(
            component,
            []
        )
    )


def get_selected_names(
    model,
    mode,
):

    if mode == "fp16":

        return set()


    if mode == "block_guided":

        selected = set()

        for name, module in (
            model.named_modules()
        ):

            if is_block_guided_target(
                name,
                module,
            ):

                selected.add(
                    name
                )

        return selected


    if mode == "act_guided":

        available = {
            name
            for name, _
            in model.named_modules()
        }


        missing = (
            ACT_GUIDED_NAMES
            - available
        )


        if missing:

            raise RuntimeError(
                "Missing activation-guided "
                f"modules: {missing}"
            )


        return set(
            ACT_GUIDED_NAMES
        )


    raise ValueError(
        f"Unknown mode: {mode}"
    )


# =============================================================================
# APPLY TERNARY CONFIGURATION
# =============================================================================

@torch.no_grad()
def apply_configuration(
    model,
    selected_names,
):

    if not selected_names:

        return {
            "matrices": 0,
            "parameters": 0,
            "mean_error": 0.0,
            "max_error": 0.0,
            "mean_scale_ratio": 0.0,
            "negative": 0,
            "zero": 0,
            "positive": 0,
        }


    matrix_count = 0
    parameter_count = 0

    weighted_error_sum = 0.0
    weighted_scale_sum = 0.0

    max_error = 0.0

    negative = 0
    zero = 0
    positive = 0


    applied_names = set()


    for name, module in (
        model.named_modules()
    ):

        if (
            name
            not in selected_names
        ):
            continue


        if not isinstance(
            module,
            torch.nn.Linear,
        ):

            raise RuntimeError(
                f"{name} is not Linear."
            )


        result = (
            optimized_ternary_weight(
                module.weight
            )
        )


        original = (
            module.weight
            .detach()
            .float()
        )


        ternary_fp32 = (
            result[
                "weight"
            ]
            .float()
        )


        difference = (
            original
            - ternary_fp32
        )


        original_norm = (
            torch.linalg.vector_norm(
                original
            ).item()
        )


        error_norm = (
            torch.linalg.vector_norm(
                difference
            ).item()
        )


        relative_error = (
            error_norm
            / original_norm
            if original_norm > 0
            else 0.0
        )


        params = (
            module.weight.numel()
        )


        module.weight.copy_(
            result["weight"]
        )


        matrix_count += 1
        parameter_count += params

        weighted_error_sum += (
            relative_error
            * params
        )

        weighted_scale_sum += (
            result["scale_ratio"]
            * params
        )

        max_error = max(
            max_error,
            relative_error,
        )


        negative += (
            result["negative"]
        )

        zero += (
            result["zero"]
        )

        positive += (
            result["positive"]
        )


        applied_names.add(
            name
        )


        del original
        del ternary_fp32
        del difference
        del result


    missing = (
        selected_names
        - applied_names
    )


    if missing:

        raise RuntimeError(
            "Selected modules were not "
            f"applied: {missing}"
        )


    return {
        "matrices":
            matrix_count,

        "parameters":
            parameter_count,

        "mean_error":
            (
                weighted_error_sum
                / parameter_count
            ),

        "max_error":
            max_error,

        "mean_scale_ratio":
            (
                weighted_scale_sum
                / parameter_count
            ),

        "negative":
            negative,

        "zero":
            zero,

        "positive":
            positive,
    }


# =============================================================================
# GENERATION
# =============================================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )


    inputs = {
        key: value.to(
            DEVICE
        )
        for key, value
        in inputs.items()
    }


    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )


    torch.cuda.synchronize()

    start = (
        time.perf_counter()
    )


    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )


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
        generated_ids.shape[0]
    )


    continuation = (
        tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
    )


    speed = (
        generated_tokens
        / elapsed
        if elapsed > 0
        else 0.0
    )


    del output
    del generated_ids

    for value in (
        inputs.values()
    ):
        del value

    del inputs


    return {
        "text":
            continuation.strip(),

        "tokens":
            generated_tokens,

        "time":
            elapsed,

        "speed":
            speed,
    }


# =============================================================================
# EXACT ANSWER SCORING
# =============================================================================

def normalize_text(text):

    text = (
        text
        .strip()
        .lower()
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def first_meaningful_line(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


    ignored = {
        "<think>",
        "</think>",
    }


    for line in lines:

        if (
            line.lower()
            not in ignored
        ):

            return line


    return ""


def score_exact_answer(
    generated_text,
    expected,
):

    if expected is None:

        return None


    expected_norm = (
        normalize_text(
            expected
        )
    )


    first_line = (
        normalize_text(
            first_meaningful_line(
                generated_text
            )
        )
    )


    # -------------------------------------------------------------------------
    # Numeric exact answers
    # -------------------------------------------------------------------------

    if re.fullmatch(
        r"-?\d+(?:\.\d+)?",
        expected_norm,
    ):

        numbers = re.findall(
            r"-?\d+(?:\.\d+)?",
            first_line,
        )


        if not numbers:

            return False


        return (
            numbers[0]
            == expected_norm
        )


    # -------------------------------------------------------------------------
    # One-word factual answers
    # -------------------------------------------------------------------------

    words = re.findall(
        r"[a-zA-Z]+",
        first_line.lower(),
    )


    if not words:

        return False


    return (
        words[0]
        == expected_norm
    )


# =============================================================================
# SIMPLE DEGENERATION DIAGNOSTICS
# =============================================================================

def repetition_score(text):

    words = re.findall(
        r"\b\w+\b",
        text.lower(),
    )


    if len(words) < 4:

        return 0.0


    bigrams = list(
        zip(
            words,
            words[1:],
        )
    )


    if not bigrams:

        return 0.0


    unique = len(
        set(
            bigrams
        )
    )


    return (
        1.0
        - (
            unique
            / len(
                bigrams
            )
        )
    )


# =============================================================================
# VALIDATION CASES
# =============================================================================

CASES = [
    {
        "name": "FP16",
        "mode": "fp16",
        "expected_matrices": 0,
        "expected_parameters": 0,
    },

    {
        "name": "BLOCK_GUIDED_7_67",
        "mode": "block_guided",
        "expected_matrices": 20,
        "expected_parameters": 57_671_680,
    },

    {
        "name": "ACT_GUIDED_7_73",
        "mode": "act_guided",
        "expected_matrices": 18,
        "expected_parameters": 58_195_968,
    },
]


# =============================================================================
# HEADER
# =============================================================================

print(
    "=" * 155
)

print(
    "EXPERIMENT 016 — "
    "FORMAL VALIDATION OF BEST "
    "TERNARY CONFIGURATIONS"
)

print(
    "=" * 155
)


print(
    "\nConfiguration"
)

print(
    "-" * 155
)


print(
    f"Model                  : "
    f"{MODEL_ID}"
)

print(
    f"Device                 : "
    f"{DEVICE}"
)

print(
    f"Data type              : "
    f"{DTYPE}"
)

print(
    f"Group size             : "
    f"{GROUP_SIZE}"
)

print(
    f"Threshold multiplier   : "
    f"{THRESHOLD_MULTIPLIER}"
)

print(
    "Reconstruction scale   : "
    "mean(abs(non-zero selected weights))"
)

print(
    f"Evaluation prompts     : "
    f"{len(EVALUATION_SET)}"
)

print(
    f"Exact-answer prompts   : "
    f"{sum(x['expected'] is not None for x in EVALUATION_SET)}"
)

print(
    f"Open-ended prompts     : "
    f"{sum(x['expected'] is None for x in EVALUATION_SET)}"
)

print(
    f"Maximum new tokens     : "
    f"{MAX_NEW_TOKENS}"
)


# =============================================================================
# ENVIRONMENT
# =============================================================================

print(
    "\nEnvironment"
)

print(
    "-" * 155
)


print(
    f"PyTorch                : "
    f"{torch.__version__}"
)

print(
    f"CUDA runtime           : "
    f"{torch.version.cuda}"
)

print(
    f"CUDA available         : "
    f"{torch.cuda.is_available()}"
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is required."
    )


print(
    f"GPU                    : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"GPU capability         : "
    f"{torch.cuda.get_device_capability(0)}"
)


total_vram = (
    torch.cuda
    .get_device_properties(0)
    .total_memory
    / (1024 ** 3)
)


print(
    f"Total VRAM             : "
    f"{total_vram:.2f} GB"
)


print(
    "\nLoading tokenizer..."
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_ID
    )
)


# =============================================================================
# RUN VALIDATION
# =============================================================================

all_case_results = []


for case_index, case in enumerate(
    CASES,
    start=1,
):

    print(
        "\n\n" + "#" * 155
    )

    print(
        f"VALIDATION CASE "
        f"{case_index}/"
        f"{len(CASES)} "
        f"— {case['name']}"
    )

    print(
        "#" * 155
    )


    model = load_model()


    total_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


    print(
        f"\nTotal model parameters : "
        f"{total_params:,}"
    )


    if (
        total_params
        != TOTAL_EXPECTED_PARAMS
    ):

        raise RuntimeError(
            "Unexpected model parameter "
            f"count: {total_params:,}"
        )


    selected_names = (
        get_selected_names(
            model,
            case["mode"],
        )
    )


    stats = (
        apply_configuration(
            model,
            selected_names,
        )
    )


    if (
        stats["matrices"]
        != case[
            "expected_matrices"
        ]
    ):

        raise RuntimeError(
            f"{case['name']} expected "
            f"{case['expected_matrices']} "
            "matrices but applied "
            f"{stats['matrices']}."
        )


    if (
        stats["parameters"]
        != case[
            "expected_parameters"
        ]
    ):

        raise RuntimeError(
            f"{case['name']} expected "
            f"{case['expected_parameters']:,} "
            "parameters but applied "
            f"{stats['parameters']:,}."
        )


    coverage = (
        stats["parameters"]
        / total_params
        * 100
    )


    total_codes = (
        stats["negative"]
        + stats["zero"]
        + stats["positive"]
    )


    if total_codes > 0:

        negative_pct = (
            stats["negative"]
            / total_codes
            * 100
        )

        zero_pct = (
            stats["zero"]
            / total_codes
            * 100
        )

        positive_pct = (
            stats["positive"]
            / total_codes
            * 100
        )

    else:

        negative_pct = 0.0
        zero_pct = 0.0
        positive_pct = 0.0


    print(
        "\nMODEL CONFIGURATION"
    )

    print(
        "-" * 155
    )


    print(
        f"Case                  : "
        f"{case['name']}"
    )

    print(
        f"Affected matrices     : "
        f"{stats['matrices']}"
    )

    print(
        f"Affected parameters   : "
        f"{stats['parameters']:,}"
    )

    print(
        f"Model coverage        : "
        f"{coverage:.2f}%"
    )

    print(
        f"Mean weight error     : "
        f"{stats['mean_error']:.6f}"
    )

    print(
        f"Max weight error      : "
        f"{stats['max_error']:.6f}"
    )

    print(
        f"Mean scale ratio      : "
        f"{stats['mean_scale_ratio']:.6f}"
    )

    print(
        f"-1 codes              : "
        f"{negative_pct:.2f}%"
    )

    print(
        f" 0 codes              : "
        f"{zero_pct:.2f}%"
    )

    print(
        f"+1 codes              : "
        f"{positive_pct:.2f}%"
    )


    # =========================================================================
    # PROMPT EVALUATION
    # =========================================================================

    exact_total = 0
    exact_correct = 0

    category_scores = {}

    total_tokens = 0
    total_time = 0.0

    repetition_scores = []

    prompt_results = []


    print(
        "\nEVALUATION"
    )

    print(
        "-" * 155
    )


    for prompt_index, item in enumerate(
        EVALUATION_SET,
        start=1,
    ):

        result = generate(
            model,
            tokenizer,
            item["prompt"],
        )


        exact_result = (
            score_exact_answer(
                result["text"],
                item["expected"],
            )
        )


        repetition = (
            repetition_score(
                result["text"]
            )
        )


        repetition_scores.append(
            repetition
        )


        if (
            item["expected"]
            is not None
        ):

            exact_total += 1


            if exact_result:

                exact_correct += 1


            category = (
                item["category"]
            )


            if (
                category
                not in category_scores
            ):

                category_scores[
                    category
                ] = {
                    "correct": 0,
                    "total": 0,
                }


            category_scores[
                category
            ][
                "total"
            ] += 1


            if exact_result:

                category_scores[
                    category
                ][
                    "correct"
                ] += 1


        total_tokens += (
            result["tokens"]
        )

        total_time += (
            result["time"]
        )


        prompt_results.append(
            {
                "id":
                    item["id"],

                "category":
                    item["category"],

                "expected":
                    item["expected"],

                "correct":
                    exact_result,

                "repetition":
                    repetition,

                "tokens":
                    result["tokens"],

                "speed":
                    result["speed"],
            }
        )


        print(
            f"\n[{item['id']}] "
            f"{item['category']}"
        )

        print(
            f"Prompt   : "
            f"{item['prompt']}"
        )


        if (
            item["expected"]
            is not None
        ):

            print(
                f"Expected : "
                f"{item['expected']}"
            )


        print(
            "\nOutput:"
        )

        print(
            result["text"]
        )


        if (
            exact_result
            is not None
        ):

            print(
                "\nExact    : "
                + (
                    "PASS"
                    if exact_result
                    else "FAIL"
                )
            )


        print(
            f"Repetition score : "
            f"{repetition:.4f}"
        )

        print(
            f"Generated tokens : "
            f"{result['tokens']}"
        )

        print(
            f"Generation time  : "
            f"{result['time']:.3f} sec"
        )

        print(
            f"Tokens/sec       : "
            f"{result['speed']:.2f}"
        )


        del result


    # =========================================================================
    # CASE SUMMARY
    # =========================================================================

    exact_accuracy = (
        exact_correct
        / exact_total
        * 100
        if exact_total > 0
        else 0.0
    )


    mean_repetition = (
        sum(
            repetition_scores
        )
        / len(
            repetition_scores
        )
    )


    throughput = (
        total_tokens
        / total_time
        if total_time > 0
        else 0.0
    )


    print(
        "\n\n" + "=" * 155
    )

    print(
        f"{case['name']} — "
        "VALIDATION SUMMARY"
    )

    print(
        "=" * 155
    )


    print(
        f"Exact-answer correct   : "
        f"{exact_correct}/"
        f"{exact_total}"
    )

    print(
        f"Exact-answer accuracy  : "
        f"{exact_accuracy:.2f}%"
    )


    for category, values in (
        category_scores.items()
    ):

        accuracy = (
            values["correct"]
            / values["total"]
            * 100
        )


        print(
            f"{category:<22}: "
            f"{values['correct']}/"
            f"{values['total']} "
            f"({accuracy:.2f}%)"
        )


    print(
        f"Mean repetition score  : "
        f"{mean_repetition:.4f}"
    )

    print(
        f"Aggregate throughput   : "
        f"{throughput:.2f} tok/s"
    )


    all_case_results.append(
        {
            "name":
                case["name"],

            "matrices":
                stats["matrices"],

            "parameters":
                stats["parameters"],

            "coverage":
                coverage,

            "mean_error":
                stats["mean_error"],

            "max_error":
                stats["max_error"],

            "exact_correct":
                exact_correct,

            "exact_total":
                exact_total,

            "exact_accuracy":
                exact_accuracy,

            "mean_repetition":
                mean_repetition,

            "throughput":
                throughput,

            "prompt_results":
                prompt_results,
        }
    )


    release_model(
        model
    )

    del model
    del stats

    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# CROSS-CONFIGURATION SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 175
)

print(
    "EXPERIMENT 016 — "
    "CROSS-CONFIGURATION SUMMARY"
)

print(
    "=" * 175
)


print(
    f"{'Configuration':<24}"
    f"{'Matrices':>10}"
    f"{'Parameters':>16}"
    f"{'Coverage':>11}"
    f"{'MeanErr':>12}"
    f"{'MaxErr':>12}"
    f"{'Exact':>12}"
    f"{'Accuracy':>12}"
    f"{'Repeat':>12}"
    f"{'Tok/s':>10}"
)


print(
    "-" * 175
)


for result in (
    all_case_results
):

    exact_string = (
        f"{result['exact_correct']}/"
        f"{result['exact_total']}"
    )


    print(
        f"{result['name']:<24}"
        f"{result['matrices']:>10}"
        f"{result['parameters']:>16,}"
        f"{result['coverage']:>10.2f}%"
        f"{result['mean_error']:>12.6f}"
        f"{result['max_error']:>12.6f}"
        f"{exact_string:>12}"
        f"{result['exact_accuracy']:>11.2f}%"
        f"{result['mean_repetition']:>12.4f}"
        f"{result['throughput']:>10.2f}"
    )


# =============================================================================
# PER-PROMPT EXACT COMPARISON
# =============================================================================

print(
    "\n\n" + "=" * 120
)

print(
    "EXACT-ANSWER PER-PROMPT COMPARISON"
)

print(
    "=" * 120
)


exact_ids = [
    item["id"]
    for item in EVALUATION_SET
    if item["expected"] is not None
]


print(
    f"{'Prompt':<10}"
    + "".join(
        f"{result['name']:>24}"
        for result
        in all_case_results
    )
)


print(
    "-" * 120
)


for prompt_id in exact_ids:

    row = (
        f"{prompt_id:<10}"
    )


    for case_result in (
        all_case_results
    ):

        matching = next(
            item
            for item
            in case_result[
                "prompt_results"
            ]
            if item["id"]
            == prompt_id
        )


        status = (
            "PASS"
            if matching[
                "correct"
            ]
            else "FAIL"
        )


        row += (
            f"{status:>24}"
        )


    print(
        row
    )


# =============================================================================
# IMPORTANT INTERPRETATION
# =============================================================================

print(
    "\n\n" + "=" * 155
)

print(
    "INTERPRETATION RULES"
)

print(
    "=" * 155
)


print(
    "1. Exact-answer accuracy is the primary "
    "automatic metric in this experiment."
)

print()

print(
    "2. Open-ended responses must not be "
    "declared correct solely from this script; "
    "they require later qualitative or "
    "benchmark-based evaluation."
)

print()

print(
    "3. Repetition score is only a degeneration "
    "diagnostic. Lower is generally preferable, "
    "but it is not a complete quality metric."
)

print()

print(
    "4. The evaluation prompts were not used "
    "for activation-guided ranking in "
    "Experiment 015."
)

print()

print(
    "5. BLOCK_GUIDED and ACT_GUIDED have "
    "approximately matched ternary coverage, "
    "making their comparison more meaningful."
)

print()

print(
    "6. Throughput must NOT be interpreted as "
    "ternary acceleration because ternary "
    "weights are reconstructed into FP16 "
    "tensors."
)

print()

print(
    "7. This experiment validates behavioral "
    "retention at ~7.7% model coverage; it "
    "does not yet validate real 2-bit storage "
    "or inference."
)


print(
    "\n" + "=" * 155
)

print(
    "EXPERIMENT 016 COMPLETE"
)

print(
    "=" * 155
)
