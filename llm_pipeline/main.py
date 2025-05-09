# llm_pipeline/main.py

import argparse
import os
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar

# Project modules
from data_utils import get_tokenizer, get_dataloader, CustomTextDataset
from model_utils import load_base_model
from base_llm_system import BaseLLMSystem

# Adaptation modules
from adaptations.custom_cuda_kernel import CustomClampedReLU # Ensure it's built
from adaptations.quantization import quantize_model_dynamic, quantize_model_static, print_size_of_model
from adaptations.lora import apply_lora as apply_lora_adaptation
from adaptations.rag import SimpleDocumentStore, RAGLLMSystem
from adaptations.flash_attention import try_load_model_with_flash_attention_2, measure_latency

# Dummy data for general training/testing
DUMMY_TRAIN_TEXTS = [
    "This is a training sentence for our LLM.",
    "PyTorch Lightning makes distributed training straightforward.",
    "We are exploring different optimization techniques.",
    "Large language models require significant computational resources.",
    "Tokenization splits text into smaller units.",
    "The quick brown fox jumps over the lazy dog.",
    "Another example text to make the dataset slightly larger.",
    "Fine-tuning adapts a pre-trained model to a specific task."
] * 20 # Multiply for a slightly larger dummy dataset

DUMMY_VAL_TEXTS = [
    "This is a validation sentence to check performance.",
    "Evaluating the model's perplexity on unseen data.",
    "Generalization is key for robust models.",
    "Let's see how well our adaptations perform."
] * 10

# Configuration (simplified defaults, could be loaded from YAML)
MODEL_NAME_DEFAULT = "gpt2" # "distilgpt2" for faster local testing
BATCH_SIZE = 2 # Keep small for local testing
MAX_LENGTH = 64
LEARNING_RATE = 1e-4 # Adjusted for small dataset/epochs
NUM_EPOCHS = 1

# Output directory
OUTPUT_DIR = "./llm_pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_trainer(experiment_name="default_experiment", adaptation_name="base"):
    """Configures and returns a PyTorch Lightning Trainer."""
    logger = TensorBoardLogger(save_dir=os.path.join(OUTPUT_DIR, "logs"), name=f"{experiment_name}/{adaptation_name}")
    
    # Check for CUDA availability for accelerator and devices
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1 # Or use 'auto' or specific number of GPUs
    else:
        accelerator = "cpu"
        devices = "auto" # PyTorch Lightning will handle CPU cores

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        log_every_n_steps=10,
        precision="bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true",
        # fast_dev_run=True # Uncomment for a very quick test run (1 batch train, 1 batch val)
    )
    return trainer

def main(args):
    print(f"Starting LLM pipeline with adaptation: {args.adaptation}")
    print(f"Using base model: {args.model_name}")
    current_time = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{args.model_name}_{current_time}"

    # 1. Load Tokenizer and Base Model (common setup)
    tokenizer = get_tokenizer(args.model_name)
    # Crucial for GPT-2 and other models that might not have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token_id})")


    # DataLoaders
    # For most adaptations (baseline, LoRA, custom CUDA kernel), we use standard dataloaders
    train_dataloader = get_dataloader(DUMMY_TRAIN_TEXTS, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, shuffle=True)
    val_dataloader = get_dataloader(DUMMY_VAL_TEXTS, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, shuffle=False)
    
    # Estimate total training steps for LR scheduler
    # Handle cases where train_dataloader might not be immediately available or used (e.g. RAG inference only)
    total_training_steps = (len(train_dataloader) * NUM_EPOCHS) if train_dataloader else 0


    # --- Apply Selected Adaptation ---
    llm_system = None
    model_to_train_or_eval = None # This will hold the model instance

    if args.adaptation == "baseline":
        print("\n--- Running Baseline Pipeline ---")
        base_model = load_base_model(args.model_name, add_pad_token=True)
        base_model.config.pad_token_id = tokenizer.pad_token_id # Ensure consistency
        model_to_train_or_eval = base_model
        llm_system = BaseLLMSystem(model_to_train_or_eval, tokenizer, learning_rate=LEARNING_RATE, total_training_steps=total_training_steps)
        trainer = get_trainer(experiment_name, "baseline")
        print("Training baseline model...")
        trainer.fit(llm_system, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        print("Evaluating baseline model...")
        trainer.test(llm_system, dataloaders=val_dataloader) # Using val_dataloader for test here

    elif args.adaptation == "custom_cuda_kernel":
        print("\n--- Running Pipeline with Custom CUDA Kernel ---")
        base_model = load_base_model(args.model_name, add_pad_token=True)
        base_model.config.pad_token_id = tokenizer.pad_token_id

        # Example: Replace a GELU activation in the first block's MLP with our custom kernel
        # This is highly model-specific. For GPT-2, it's usually in `model.transformer.h[idx].mlp.act`
        try:
            if hasattr(base_model, 'transformer') and \
               len(base_model.transformer.h) > 0 and \
               hasattr(base_model.transformer.h[0], 'mlp') and \
               hasattr(base_model.transformer.h[0].mlp, 'act'):

                original_activation = base_model.transformer.h[0].mlp.act.__class__.__name__
                base_model.transformer.h[0].mlp.act = CustomClampedReLU(clamp_val=5.0) # From config or default
                print(f"Replaced activation '{original_activation}' in the first block's MLP with CustomClampedReLU.")
                model_to_train_or_eval = base_model
            else:
                print("Could not find standard MLP activation to replace for GPT-2 structure. Using base model.")
                model_to_train_or_eval = base_model
        except Exception as e:
            print(f"Error modifying model for custom CUDA kernel: {e}. Using base model.")
            model_to_train_or_eval = base_model
        
        llm_system = BaseLLMSystem(model_to_train_or_eval, tokenizer, learning_rate=LEARNING_RATE, total_training_steps=total_training_steps)
        trainer = get_trainer(experiment_name, "custom_cuda")
        print("Training model with custom CUDA kernel...")
        trainer.fit(llm_system, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        print("Evaluating model with custom CUDA kernel...")
        trainer.test(llm_system, dataloaders=val_dataloader)

    elif args.adaptation == "quantization":
        print("\n--- Running Pipeline with Quantization ---")
        base_model_fp32 = load_base_model(args.model_name, add_pad_token=True) # Load fresh for quantization
        base_model_fp32.config.pad_token_id = tokenizer.pad_token_id
        base_model_fp32.eval() # Quantization is usually done on an eval model

        print_size_of_model(base_model_fp32, "FP32")
        
        device_for_quant = 'cpu' # Quantization often targets CPU deployment
        base_model_fp32.to(device_for_quant)

        # For static quantization, we need a calibration dataloader
        # Create a small calibration dataset, ensuring correct input format for the model
        calibration_texts = DUMMY_TRAIN_TEXTS[:32] # Use a subset for calibration
        # For calibration, ensure your CustomTextDataset returns a dictionary
        # or that your calibrate_model function handles the model's expected input format.
        # The `calibrate_model` in `quantization.py` expects dicts from dataloader
        calibration_dataloader = get_dataloader(calibration_texts, tokenizer, batch_size=4, max_length=MAX_LENGTH, shuffle=False)


        quantized_model = None
        if args.quant_type == "dynamic":
            print("Applying Dynamic Quantization...")
            # For GPT2, common layers are nn.Linear. Some submodules might be Conv1D which acts like Linear.
            # PEFT's `get_peft_model` might change module types, so inspect model carefully.
            # For a standard HuggingFace GPT2Model, `torch.nn.Linear` is a good target for dynamic quantization.
            # If you use `Conv1D` layers from HF that are effectively Linear, you might need to add them.
            quantized_model = quantize_model_dynamic(base_model_fp32, modules_to_quantize={torch.nn.Linear})
        elif args.quant_type == "static":
            print("Applying Static Quantization...")
            print("Note: Static quantization of full transformers can be complex and error-prone with basic PyTorch API.")
            print("Hugging Face Optimum or more tailored approaches are often better.")
            try:
                # Ensure the model is on CPU for static quantization preparation
                quantized_model = quantize_model_static(base_model_fp32, calibration_dataloader, device=device_for_quant)
            except Exception as e:
                print(f"Static Quantization failed: {e}. This is common for complex models.")
                print("Consider using HuggingFace Optimum or focusing on Dynamic Quantization / QAT for transformers.")
                quantized_model = None # Fallback or error

        if quantized_model:
            print_size_of_model(quantized_model, f"{args.quant_type.capitalize()} Quantized")
            model_to_train_or_eval = quantized_model # For evaluation
            model_to_train_or_eval.to(device_for_quant) # Ensure it's on the correct device for eval

            # Evaluate the quantized model (e.g., perplexity)
            # We create a BaseLLMSystem but won't train it.
            # The loss calculation needs to work on CPU for the quantized model.
            # Ensure BaseLLMSystem handles device placement correctly or move data to CPU.
            quant_llm_system = BaseLLMSystem(model_to_train_or_eval, tokenizer)
            
            # Create a new trainer for CPU evaluation if needed
            trainer_quant = pl.Trainer(
                accelerator="cpu", devices=1, logger=False, # No need to log extensively for this eval
                callbacks=[TQDMProgressBar(refresh_rate=1)]
            )
            print(f"Evaluating {args.quant_type} quantized model on CPU...")
            
            # Adjust dataloader for CPU eval if necessary (data already on CPU from get_dataloader by default)
            # Ensure batch items are moved to cpu if BaseLLMSystem expects it.
            # The common_step in BaseLLMSystem should handle data on the model's device.
            trainer_quant.test(quant_llm_system, dataloaders=val_dataloader) # Using val_dataloader
            
            # Latency test (simple)
            test_input_text = "Evaluate quantized model latency."
            encoded_input = tokenizer(test_input_text, return_tensors='pt').to(device_for_quant)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model_to_train_or_eval(**encoded_input)
            
            start_time = time.time()
            num_runs = 20
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model_to_train_or_eval(**encoded_input)
            avg_latency = (time.time() - start_time) / num_runs * 1000 # ms
            print(f"Average inference latency ({args.quant_type} quantized on CPU): {avg_latency:.2f} ms")
        else:
            print(f"{args.quant_type.capitalize()} quantization was not successful or skipped.")


    elif args.adaptation == "lora":
        print("\n--- Running Pipeline with LoRA ---")
        base_model = load_base_model(args.model_name, add_pad_token=True)
        base_model.config.pad_token_id = tokenizer.pad_token_id

        # For GPT-2, target_modules like "c_attn" are common for attention qkv.
        # "c_proj" for attention output, "c_fc" for MLP feedforward.
        # PEFT usually handles finding these if you provide parts of the name.
        lora_target_modules = ["c_attn"] # Default for GPT-2, can be configured
        if "gpt2" not in args.model_name.lower(): # Crude check, ideally use model config
            print(f"Warning: Default LoRA target_modules {lora_target_modules} are for GPT-2. May need adjustment for {args.model_name}")
        
        lora_model, _ = apply_lora_adaptation(
            base_model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            task_type="CAUSAL_LM" # For GPT-2 like models
        )
        model_to_train_or_eval = lora_model
        
        # Create LLM system. Optimizer in BaseLLMSystem will only pick up trainable (LoRA) params.
        llm_system = BaseLLMSystem(model_to_train_or_eval, tokenizer, learning_rate=LEARNING_RATE * 2, total_training_steps=total_training_steps) # May need different LR for LoRA
        trainer = get_trainer(experiment_name, "lora")
        print("Fine-tuning with LoRA...")
        trainer.fit(llm_system, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        print("Evaluating LoRA model...")
        trainer.test(llm_system, dataloaders=val_dataloader)
        
        # Save LoRA adapter
        adapter_path = os.path.join(OUTPUT_DIR, f"{args.model_name}_lora_adapters_{current_time}")
        model_to_train_or_eval.save_pretrained(adapter_path)
        print(f"LoRA adapter saved to {adapter_path}")


    elif args.adaptation == "rag":
        print("\n--- Running Pipeline with RAG (Inference Demo) ---")
        base_model = load_base_model(args.model_name, add_pad_token=True)
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.eval() # RAG usually for inference with a pre-trained LLM

        rag_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(rag_device)

        # Sample documents for RAG store
        sample_docs_for_rag = [
            "The capital of France is Paris, known for the Eiffel Tower.",
            "PyTorch is a popular deep learning framework created by Facebook.",
            "Mount Everest is the highest peak in the world, part of the Himalayas.",
            "The first human to walk on the moon was Neil Armstrong in 1969.",
            "Photosynthesis in plants converts light energy into chemical energy."
        ] * 2 # A few more docs
        
        doc_store = SimpleDocumentStore(documents=sample_docs_for_rag)
        rag_system = RAGLLMSystem(
            llm_model=base_model,
            llm_tokenizer=tokenizer,
            document_store=doc_store,
            k_retrieval=2
        )

        prompts = [
            "What is the capital of France?",
            "Who made PyTorch?",
            "How do plants get energy?"
        ]
        for prompt in prompts:
            print(f"\nRAG Query: {prompt}")
            # The RAGLLMSystem's generate method handles retrieval and augmented prompt generation
            answer, retrieved_docs, augmented_prompt = rag_system.generate(
                prompt,
                max_new_tokens=50,
                num_beams=3, # Example generation parameters
                early_stopping=True
            )
            print(f"Retrieved context (first doc): {retrieved_docs[0]['text'][:100]}...")
            # print(f"Augmented Prompt used: {augmented_prompt}") # Can be very long
            print(f"RAG Answer: {answer}")
        model_to_train_or_eval = base_model # Just to have something assigned, not trained here

    elif args.adaptation == "flash_attention":
        print("\n--- Running Pipeline with Flash Attention (Inference Demo) ---")
        fa_model_name = args.flash_attention_model_name if args.flash_attention_model_name else args.model_name
        print(f"Attempting to use model: {fa_model_name} for Flash Attention test.")

        if not torch.cuda.is_available():
            print("Flash Attention requires CUDA. Skipping this adaptation.")
            return

        # Tokenizer for the FA model
        fa_tokenizer = get_tokenizer(fa_model_name)
        if fa_tokenizer.pad_token is None:
            fa_tokenizer.pad_token = fa_tokenizer.eos_token

        model_fa, status_fa = try_load_model_with_flash_attention_2(fa_model_name, trust_remote_code=args.trust_remote_code)

        if model_fa:
            print(f"Model ({fa_model_name}) loaded with status: {status_fa}")
            latency_fa = measure_latency(model_fa, fa_tokenizer, device="cuda", num_runs=20, max_length=100)
            print(f"Average inference latency with potential Flash Attention ({fa_model_name}): {latency_fa:.2f} ms")
            
            # Example generation
            prompt = "Flash Attention is designed to "
            inputs = fa_tokenizer(prompt, return_tensors="pt").to(model_fa.device)
            with torch.no_grad():
                generated_ids = model_fa.generate(inputs.input_ids, max_new_tokens=30, pad_token_id=fa_tokenizer.pad_token_id)
            generated_text = fa_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Generated text from FA model: {generated_text}")

            del model_fa # Free memory
            torch.cuda.empty_cache()
            model_to_train_or_eval = "FlashAttentionModel (not stored)" # Placeholder
        else:
            print(f"Failed to load model {fa_model_name} with Flash Attention. Status: {status_fa}")
            print("Skipping Flash Attention specific parts.")
            return # Exit if FA model fails to load for this demo

    else:
        print(f"Unknown adaptation: {args.adaptation}")
        return

    # Common: Basic Prediction/Generation (if a model was trained/loaded and it's a CausalLM)
    if model_to_train_or_eval and hasattr(model_to_train_or_eval, 'generate') and args.adaptation not in ["rag", "quantization", "flash_attention"]: # RAG/Quant/FA have their own demo
        print("\n--- Performing a quick generation test ---")
        prompt_text = "Once upon a time"
        # Ensure model is on the correct device for generation if not already handled by Trainer
        device = next(model_to_train_or_eval.parameters()).device
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        model_to_train_or_eval.eval() # Set to eval mode for generation
        with torch.no_grad():
            output_ids = model_to_train_or_eval.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=30,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=3, # Example: use beam search
                early_stopping=True
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Prompt: '{prompt_text}'")
        print(f"Generated text: '{generated_text}'")

    print(f"\nPipeline run for adaptation '{args.adaptation}' finished.")
    print(f"Logs and outputs (if any) are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular LLM Pipeline with PyTorch Lightning")
    parser.add_argument(
        "--adaptation",
        type=str,
        default="baseline",
        choices=["baseline", "custom_cuda_kernel", "quantization", "lora", "rag", "flash_attention"],
        help="Which adaptation to run."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME_DEFAULT,
        help="Base Hugging Face model name (e.g., gpt2, distilgpt2)."
    )
    # Quantization specific args
    parser.add_argument(
        "--quant_type",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Type of post-training quantization (if adaptation is 'quantization')."
    )
    # LoRA specific args
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank r.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    # Flash Attention specific args
    parser.add_argument(
        "--flash_attention_model_name",
        type=str,
        default="EleutherAI/gpt-neo-125M", # A model more likely to support FA out-of-box
        help="Model name to try with Flash Attention (if different from main model_name)."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow trusting remote code for Hugging Face model loading (e.g., for some Flash Attention models)."
    )


    # CUDA Kernel build check (manual reminder)
    try:
        from adaptations.custom_cuda_kernel.kernel_wrapper import custom_cuda_kernels_cpp
        if custom_cuda_kernels_cpp is None and "custom_cuda_kernel" in parser.parse_known_args()[0].adaptation :
             print("\nWARNING: Custom CUDA kernel does not seem to be compiled.")
             print("Please navigate to 'llm_pipeline/adaptations/custom_cuda_kernel/' and run 'python setup.py build_ext --inplace'\n")
    except ImportError:
        if "custom_cuda_kernel" in parser.parse_known_args()[0].adaptation:
            print("\nWARNING: Custom CUDA kernel module not found. It might not be compiled.")
            print("Please navigate to 'llm_pipeline/adaptations/custom_cuda_kernel/' and run 'python setup.py build_ext --inplace'\n")


    cli_args = parser.parse_args()
    main(cli_args)