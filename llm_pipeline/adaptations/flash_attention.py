# llm_pipeline/adaptations/flash_attention.py
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import time

# Placeholder for measuring inference latency (simplified)
def measure_latency(model, tokenizer, text="Hello world", device="cuda", num_runs=10, max_length=50):
    model.to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Warm-up runs
    for _ in range(5):
        with torch.no_grad():
            _ = model.generate(inputs.input_ids, max_new_tokens=max_length, pad_token_id=tokenizer.pad_token_id)
    
    if device == "cuda": torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.generate(inputs.input_ids, max_new_tokens=max_length, pad_token_id=tokenizer.pad_token_id)
    
    if device == "cuda": torch.cuda.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs * 1000  # milliseconds
    return avg_latency

def try_load_model_with_flash_attention_2(model_name: str, add_pad_token: bool = True, trust_remote_code: bool = True):
    """
    Attempts to load a model using Flash Attention 2 via the attn_implementation argument.
    Requires transformers >= 4.31 and flash-attn library installed, and a compatible GPU.
    """
    if not torch.cuda.is_available():
        print("Flash Attention 2 requires a CUDA-enabled GPU. Skipping.")
        return None, "No CUDA GPU available"

    try:
        # Ensure flash-attn is installed and accessible
        import flash_attn
        print(f"Flash Attention library found (version: {flash_attn.__version__}).")
    except ImportError:
        print("Flash Attention (flash-attn) library not found. Please install it: pip install flash-attn --no-build-isolation")
        return None, "flash-attn not installed"

    try:
        print(f"Attempting to load {model_name} with attn_implementation='flash_attention_2'...")
        # Some models might require trust_remote_code=True if they have custom code for FlashAttention
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, # FlashAttention often works best with bfloat16 or float16
            trust_remote_code=trust_remote_code
        ).to("cuda")
        print(f"Successfully loaded {model_name} with Flash Attention 2.")
        
        # Verify if FlashAttention is actually being used (heuristic)
        uses_fa = False
        for name, module in model.named_modules():
            if "FlashAttention" in module.__class__.__name__ or "FlashSelfAttention" in module.__class__.__name__:
                print(f"Found Flash Attention layer: {name} ({module.__class__.__name__})")
                uses_fa = True
                break
        if not uses_fa:
            print("Warning: Model loaded, but could not confirm Flash Attention layers in model structure. It might be a fallback or not fully supported by this model version with this method.")
            
        return model, "Loaded with flash_attention_2 (best effort check)"
    
    except Exception as e:
        print(f"Failed to load model with Flash Attention 2 directly: {e}")
        print("This could be due to: model not supporting it, older transformers version, flash-attn not properly installed, or GPU incompatibility.")
        print("You can also try 'sdpa' (Scaled Dot Product Attention) as attn_implementation if PyTorch >= 2.0.")
        
        try:
            print(f"Attempting to load {model_name} with attn_implementation='sdpa' (PyTorch native scaled_dot_product_attention)...")
            # SDPA is PyTorch's native implementation and can be accelerated on compatible hardware.
            # It might use FlashAttention kernels under the hood if flash-attn is installed and hardware is compatible.
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code
            ).to("cuda")
            print(f"Successfully loaded {model_name} with SDPA.")
            # Check if SDPA is using FlashAttention backend
            # This is harder to check programmatically from model structure alone
            # but PyTorch will attempt to use efficient backends.
            return model, "Loaded with sdpa"
        except Exception as e_sdpa:
            print(f"Failed to load model with SDPA: {e_sdpa}")
            return None, f"Failed: {e}"


if __name__ == "__main__":
    from ..model_utils import load_base_model
    from ..data_utils import get_tokenizer

    # Use a model known to be compatible with Flash Attention for a better demo,
    # e.g., newer versions of Llama, Falcon, or some GPT-NeoX models.
    # GPT-2 itself might not directly support the `attn_implementation` flag as easily as newer models.
    # Let's try with a common model like "EleutherAI/gpt-neo-125M" or "tiiuae/falcon-7b" (if resources allow)
    # For a smaller test, "distilgpt2" or "gpt2" can be used to see if SDPA works.
    
    # MODEL_NAME_FA = "EleutherAI/gpt-neo-125M" # Known to work well with FA optimizations
    MODEL_NAME_FA = "gpt2" # Let's try with gpt2 for consistency and see if SDPA path works
                           # Or if FlashAttention-2 path works with newer transformers.

    tokenizer_fa = get_tokenizer(MODEL_NAME_FA)
    if tokenizer_fa.pad_token is None:
        tokenizer_fa.pad_token = tokenizer_fa.eos_token

    if torch.cuda.is_available():
        print(f"\n--- Testing Flash Attention for {MODEL_NAME_FA} ---")
        
        # Attempt to load with Flash Attention 2
        model_fa, status_fa = try_load_model_with_flash_attention_2(MODEL_NAME_FA)

        if model_fa:
            print(f"\nModel ({MODEL_NAME_FA}) status: {status_fa}")
            latency_fa = measure_latency(model_fa, tokenizer_fa, device="cuda")
            print(f"Average inference latency with potential Flash Attention: {latency_fa:.2f} ms")
            del model_fa # Free memory
            torch.cuda.empty_cache()
        else:
            print(f"Could not load {MODEL_NAME_FA} with Flash Attention or SDPA. Status: {status_fa}")

        # Compare with standard attention model
        print(f"\n--- Loading {MODEL_NAME_FA} with standard attention for comparison ---")
        try:
            # Load with torch_dtype=torch.bfloat16 for a fairer comparison if FA model used it
            model_standard = AutoModelForCausalLM.from_pretrained(MODEL_NAME_FA, torch_dtype=torch.bfloat16).to("cuda")
            # Or if bfloat16 causes issues for the standard model on some hardware, use float32 or float16
            # model_standard = load_base_model(MODEL_NAME_FA).to("cuda") # from our model_utils
            
            print(f"Successfully loaded {MODEL_NAME_FA} with standard attention.")
            latency_standard = measure_latency(model_standard, tokenizer_fa, device="cuda")
            print(f"Average inference latency with standard attention: {latency_standard:.2f} ms")
            del model_standard
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading/testing standard model {MODEL_NAME_FA}: {e}")

    else:
        print("CUDA not available, skipping Flash Attention tests.")

    print("\nNote: Flash Attention benefits are most prominent on specific GPU architectures (Ampere, Hopper)")
    print("and with longer sequences. Ensure 'flash-attn' is correctly installed for your CUDA and PyTorch version.")
    print("For `attn_implementation`, `transformers` library should be up-to-date.")