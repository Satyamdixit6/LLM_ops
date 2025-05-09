# llm_pipeline/adaptations/quantization.py
import torch
import torch.quantization as tq
from torch.quantization import get_default_qconfig, quantize_jit
from copy import deepcopy
import time
import os

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp_model.p")
    size = os.path.getsize("temp_model.p")
    print(f"Size ({label}): {size/1e6:.2f} MB")
    os.remove("temp_model.p")
    return size

def calibrate_model(model, dataloader, device='cpu'):
    """Calibrates the model using the provided dataloader."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Ensure the model's forward pass is called for calibration
            # For HuggingFace models, this often means just calling the model
            try:
                model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e:
                # Some models might require specific input structures for scripting/tracing
                # This is a common issue with quantization.
                # For GPT2, a tuple of (input_ids, attention_mask) might be expected by a scripted model.
                try:
                    model(input_ids, attention_mask) # Try with tuple directly if forward is overloaded
                except Exception as e2:
                    print(f"Calibration failed: {e2}. Ensure model's forward pass is compatible with scripting.")
                    raise

def quantize_model_static(model_fp32, dataloader_for_calibration, qconfig_str='fbgemm', device='cpu'):
    """
    Applies post-training static quantization to the model.
    Returns the quantized model.
    """
    model_to_quantize = deepcopy(model_fp32)
    model_to_quantize.eval()
    model_to_quantize.to(device) # Quantization is typically CPU-focused for QAT or static.

    # 1. Specify quantization configuration
    # 'fbgemm' for x86, 'qnnpack' for ARM.
    # Ensure backend is supported, otherwise use default.
    if device == 'cpu': # Default qconfig is for CPU
        if qconfig_str not in torch.backends.quantized.supported_engines:
            print(f"Warning: {qconfig_str} not in supported_engines: {torch.backends.quantized.supported_engines}. Using default.")
            qconfig_str = torch.backends.quantized.engine # Default engine
    
    if qconfig_str == 'qnnpack' and 'qnnpack' not in torch.backends.quantized.supported_engines:
        print("Warning: qnnpack not available, defaulting to fbgemm or other available engine.")
        qconfig_str = 'fbgemm' # or some other default
    
    torch.backends.quantized.engine = qconfig_str
    model_to_quantize.qconfig = get_default_qconfig(qconfig_str)
    print(f"Using qconfig: {model_to_quantize.qconfig} with engine {qconfig_str}")


    # 2. Fuse modules: Combine layers like Conv-BN-ReLU
    # For transformers, fusion might be less common or automatic with prepare.
    # Example: torch.quantization.fuse_modules(model_to_quantize, [['conv', 'relu']], inplace=True)
    # For transformers, this step might be skipped or handled by `prepare`.
    # We need to be careful about what can be fused in a transformer.
    # Often, linear layers and activations are the primary targets.
    print("Preparing model for quantization...")
    torch.quantization.prepare(model_to_quantize, inplace=True)

    # 3. Calibrate the model
    print("Calibrating model...")
    calibrate_model(model_to_quantize, dataloader_for_calibration, device=device)
    print("Calibration complete.")

    # 4. Convert the model to a quantized version
    print("Converting model to quantized version...")
    model_quantized = torch.quantization.convert(model_to_quantize, inplace=True)
    model_quantized.eval()
    print("Quantization complete.")

    return model_quantized

def quantize_model_dynamic(model_fp32, qconfig_str='fbgemm', modules_to_quantize=None):
    """
    Applies post-training dynamic quantization.
    Modules_to_quantize: set of nn.Module types to quantize, e.g., {nn.Linear, nn.LSTM}
    """
    if modules_to_quantize is None:
        modules_to_quantize = {torch.nn.Linear} # Default for transformers

    model_to_quantize = deepcopy(model_fp32)
    model_to_quantize.eval()
    # Dynamic quantization happens on CPU typically
    model_to_quantize.to('cpu')
    
    if qconfig_str not in torch.backends.quantized.supported_engines:
        print(f"Warning: {qconfig_str} not in supported_engines. Using default.")
        qconfig_str = torch.backends.quantized.engine
    torch.backends.quantized.engine = qconfig_str
    
    model_quantized = torch.quantization.quantize_dynamic(
        model_to_quantize,
        qconfig_spec=modules_to_quantize, # Specify which module types to quantize
        dtype=torch.qint8 # Target data type for weights
    )
    print("Dynamic quantization complete.")
    return model_quantized


if __name__ == "__main__":
    from ..model_utils import load_base_model # Use .. for relative import when running as part of package
    from ..data_utils import get_tokenizer, get_dataloader

    # Setup: Load a pre-trained model (e.g., DistilGPT2 or a small BERT)
    # Using a smaller model for quicker testing of quantization.
    MODEL_NAME = "distilgpt2" # GPT2 can also be used but is larger
    model_fp32 = load_base_model(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)

    dummy_texts = [
        "This is a sentence for calibration.",
        "Quantization helps reduce model size.",
        "We need a representative dataset for calibration."
    ] * 10 # Small dataset for calibration demo

    # Prepare a dataloader for calibration (only for static quantization)
    # Use a small max_length for faster calibration in this example
    calibration_dataloader = get_dataloader(dummy_texts, tokenizer, batch_size=2, max_length=32, shuffle=False)

    print_size_of_model(model_fp32, "FP32")

    # --- Test Static Quantization ---
    print("\n--- Testing Static Quantization (CPU) ---")
    # Static quantization requires model to be on CPU and often specific qconfig
    # Note: Full transformer quantization can be tricky and might require model surgery
    # or specific support (e.g., Hugging Face Optimum for some models).
    # Here we try a basic approach.
    try:
        # Static quantization works best on CPU with 'fbgemm' or 'qnnpack'
        # It often requires the model to be scriptable or traceable, which can be an issue for complex Python models.
        # For demonstration, we'll proceed, but be aware it might fail for some transformer architectures out-of-the-box.
        # We'll try to make the model compatible if simple changes are needed.
        
        # Workaround for GPT2: It may not be directly scriptable for all quantization paths.
        # One common approach is to quantize parts of the model or use specific tools.
        # For a simple demo, let's try dynamic quantization first as it's more forgiving.
        
        # model_static_quantized = quantize_model_static(model_fp32, calibration_dataloader, qconfig_str='fbgemm', device='cpu')
        # print_size_of_model(model_static_quantized, "Static Quantized (CPU)")

        # # Test inference (example)
        # model_static_quantized.eval()
        # test_input_text = "Hello, quantized world!"
        # encoded_input = tokenizer(test_input_text, return_tensors='pt').to('cpu')
        # with torch.no_grad():
        #     output_static = model_static_quantized(**encoded_input)
        # print("Static quantized model output (logits shape):", output_static.logits.shape)
        print("Static quantization for complex transformers like GPT-2 can be challenging with basic PyTorch API.")
        print("It often requires specific model adjustments or tools like Hugging Face Optimum.")
        print("Skipping direct static quantization test in this generic script due to potential compatibility issues.")
        print("Focusing on dynamic quantization which is generally more applicable.")


    except Exception as e:
        print(f"Static Quantization failed: {e}")
        print("This can happen if the model isn't easily convertible (e.g., not scriptable, unsupported ops).")

    # --- Test Dynamic Quantization ---
    print("\n--- Testing Dynamic Quantization (CPU) ---")
    # Dynamic quantization is simpler and applied to specified layers (e.g., Linear)
    try:
        # Specify layers common in transformers. Add others if needed.
        modules_to_dynamically_quantize = {torch.nn.Linear, torch.nn.Embedding} # Embedding might be tricky
        modules_to_dynamically_quantize = {torch.nn.Linear} # Safer bet for general transformers

        model_dynamic_quantized = quantize_model_dynamic(model_fp32, qconfig_str='fbgemm', modules_to_quantize=modules_to_dynamically_quantize)
        print_size_of_model(model_dynamic_quantized, "Dynamic Quantized (CPU)")

        # Test inference
        model_dynamic_quantized.eval() # Already in eval from quantize function
        test_input_text = "Hello, dynamically quantized world!"
        # For dynamic quantization, ensure the model is on CPU and inputs match.
        encoded_input = tokenizer(test_input_text, return_tensors='pt').to('cpu')
        with torch.no_grad():
            output_dynamic = model_dynamic_quantized(**encoded_input)
        print("Dynamic quantized model output (logits shape):", output_dynamic.logits.shape)

        # You can also try to generate text (if it's a CausalLM)
        if hasattr(model_dynamic_quantized, 'generate'):
            print("Generating text with dynamically quantized model...")
            generated_ids = model_dynamic_quantized.generate(
                encoded_input['input_ids'],
                max_length=20,
                pad_token_id=tokenizer.pad_token_id
            )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Generated text: {generated_text}")

    except Exception as e:
        print(f"Dynamic Quantization failed: {e}")

    # Note: For full transformer quantization, especially with good accuracy,
    # Quantization Aware Training (QAT) or specialized tools like Hugging Face Optimum
    # are often preferred over basic post-training static/dynamic quantization.