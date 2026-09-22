import time
import torch

from transformers import AutoTokenizer, Qwen3_5ForCausalLM


MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEVICE = "cuda"
DTYPE = torch.float16


def parameter_memory_gb(model):
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 3)


def mb(value):
    return value / (1024 ** 2)


print("=" * 70)
print("EXPERIMENT 001B — Qwen3.5-0.8B GPU FP16 Baseline")
print("=" * 70)

print("\nEnvironment")
print("-" * 70)

print("PyTorch        :", torch.__version__)
print("CUDA runtime   :", torch.version.cuda)
print("CUDA available :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU            :", torch.cuda.get_device_name(0))
print("Capability     :", torch.cuda.get_device_capability(0))

gpu_properties = torch.cuda.get_device_properties(0)

print(
    f"Total VRAM     : "
    f"{gpu_properties.total_memory / (1024 ** 3):.2f} GB"
)

# ---------------------------------------------------------
# Reset memory measurements
# ---------------------------------------------------------

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

print("Loading FP16 model...")

start = time.time()

model = Qwen3_5ForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
)

model = model.to(DEVICE)
model.eval()

torch.cuda.synchronize()

load_time = time.time() - start

# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

total_params = sum(
    p.numel()
    for p in model.parameters()
)

allocated_after_load = torch.cuda.memory_allocated()
reserved_after_load = torch.cuda.memory_reserved()

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(f"Model              : {MODEL_ID}")
print(f"Device             : {DEVICE}")
print(f"Data type          : {DTYPE}")
print(f"Parameters         : {total_params:,}")
print(f"Parameter memory   : {parameter_memory_gb(model):.3f} GB")
print(f"Load time          : {load_time:.2f} sec")

print(
    f"GPU allocated      : "
    f"{mb(allocated_after_load):.2f} MB"
)

print(
    f"GPU reserved       : "
    f"{mb(reserved_after_load):.2f} MB"
)

# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

prompt = "Explain in simple terms why the sky is blue."

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}

# ---------------------------------------------------------
# Generation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("GENERATION")
print("=" * 70)

print("\nPrompt:")
print(prompt)

# Reset peak so we measure generation separately.
torch.cuda.reset_peak_memory_stats()

torch.cuda.synchronize()
start = time.time()

with torch.inference_mode():

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )

torch.cuda.synchronize()

generation_time = time.time() - start

peak_generation_memory = torch.cuda.max_memory_allocated()

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

input_tokens = inputs["input_ids"].shape[-1]

generated_tokens = (
    outputs.shape[-1] - input_tokens
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)

print("\nResponse:")
print(response)

# ---------------------------------------------------------
# Performance
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PERFORMANCE")
print("=" * 70)

print(f"Input tokens        : {input_tokens}")
print(f"Generated tokens    : {generated_tokens}")
print(f"Generation time     : {generation_time:.2f} sec")

if generation_time > 0:

    tokens_per_second = (
        generated_tokens / generation_time
    )

    print(
        f"Tokens/sec          : "
        f"{tokens_per_second:.2f}"
    )

print(
    f"Peak GPU memory     : "
    f"{mb(peak_generation_memory):.2f} MB"
)

print(
    f"Current allocated   : "
    f"{mb(torch.cuda.memory_allocated()):.2f} MB"
)

print(
    f"Current reserved    : "
    f"{mb(torch.cuda.memory_reserved()):.2f} MB"
)
