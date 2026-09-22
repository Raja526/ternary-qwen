import time
import torch

from transformers import AutoTokenizer, Qwen3_5ForCausalLM


MODEL_ID = "Qwen/Qwen3.5-0.8B"

# Start conservatively on CPU.
# After this works, we will benchmark the P1000 separately.
DEVICE = "cpu"
DTYPE = torch.float32


def parameter_memory(model):
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    return total_bytes / (1024 ** 3)


print("=" * 70)
print("EXPERIMENT 001 — Qwen3.5-0.8B Baseline")
print("=" * 70)

print("\nEnvironment")
print("-" * 70)
print("PyTorch       :", torch.__version__)
print("CUDA runtime  :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU           :", torch.cuda.get_device_name(0))
    print("GPU capability:", torch.cuda.get_device_capability(0))

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")

start = time.time()

model = Qwen3_5ForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
)

model = model.to(DEVICE)
model.eval()

load_time = time.time() - start


# ---------------------------------------------------------
# Model statistics
# ---------------------------------------------------------

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(f"Model             : {MODEL_ID}")
print(f"Device            : {DEVICE}")
print(f"Data type         : {DTYPE}")
print(f"Parameters        : {total_params:,}")
print(f"Trainable params  : {trainable_params:,}")
print(f"Parameter memory  : {parameter_memory(model):.3f} GB")
print(f"Load time         : {load_time:.2f} sec")


# ---------------------------------------------------------
# Generation
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


print("\n" + "=" * 70)
print("GENERATION")
print("=" * 70)

print("\nPrompt:")
print(prompt)

start = time.time()

with torch.inference_mode():

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )

generation_time = time.time() - start


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

print(f"Input tokens     : {input_tokens}")
print(f"Generated tokens : {generated_tokens}")
print(f"Generation time  : {generation_time:.2f} sec")

if generation_time > 0:

    tokens_per_second = (
        generated_tokens /
        generation_time
    )

    print(
        f"Tokens/sec       : "
        f"{tokens_per_second:.2f}"
    )
