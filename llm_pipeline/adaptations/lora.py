# llm_pipeline/adaptations/lora.py

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

def apply_lora(model, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None, task_type=TaskType.CAUSAL_LM):
    """
    Applies LoRA to the given model.

    Args:
        model: The Hugging Face transformer model.
        r (int): LoRA rank.
        lora_alpha (int): LoRA alpha scaling factor.
        lora_dropout (float): Dropout probability for LoRA layers.
        target_modules (list of str, optional):
            List of module names to apply LoRA to (e.g., ["q_proj", "v_proj"]).
            If None, PEFT will try to infer common targets for the model architecture.
            For GPT2, common targets are often 'c_attn' or breaking it down if the library expects individual query/key/value.
            PEFT often handles this well if target_modules is not specified or correctly specified.
            For GPT2, typical targets are like `['c_attn']` or sometimes it could be more granular depending on the peft version and model implementation.
            Let's try with a common one for GPT2 like 'c_attn' or let peft decide.
            It's often better to look up recommended `target_modules` for specific model architectures.
            For `transformers.GPT2Model`, 'c_attn' is a common target for Q, K, V projections combined.
        task_type (peft.TaskType): The task type for PEFT.
    Returns:
        model: The PEFT model with LoRA layers.
        lora_config: The LoraConfig object.
    """

    # If target_modules are not specified, PEFT might make a good guess for some models.
    # For GPT2, 'c_attn' is a common target for the attention projection layer.
    # If you were using a model like Llama, target_modules might be ["q_proj", "k_proj", "v_proj", "o_proj"].
    if target_modules is None:
        if "gpt2" in model.config.model_type.lower():
            target_modules = ["c_attn"] # This is a common one for GPT-2's attention projections.
                                        # Sometimes 'c_proj' for output projection and 'c_fc' for MLP layers are also targeted.
        # Add other model-specific defaults if needed
        # else:
        #     print("Warning: target_modules not specified for LoRA. PEFT will attempt to infer.")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",  # or 'all' or 'lora_only'. 'none' is common.
        task_type=task_type,
        # modules_to_save = ["lm_head"] # Example: if you want to train the lm_head as well
    )

    # Prepare model for k-bit training if you were doing QLoRA (e.g., 4-bit or 8-bit)
    # For standard LoRA on a non-quantized model, this might not be strictly necessary
    # but can be good practice if you might later quantize the base.
    # model = prepare_model_for_kbit_training(model) # Uncomment if using k-bit quantization for base model

    peft_model = get_peft_model(model, lora_config)
    print("LoRA applied successfully.")
    peft_model.print_trainable_parameters()

    return peft_model, lora_config

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

if __name__ == "__main__":
    from ..model_utils import load_base_model # Relative import
    from ..data_utils import get_tokenizer, get_dataloader
    from ..base_llm_system import BaseLLMSystem
    import pytorch_lightning as pl

    MODEL_NAME = "gpt2" # "distilgpt2" for faster testing
    tokenizer = get_tokenizer(MODEL_NAME)
    base_model = load_base_model(MODEL_NAME, add_pad_token=True) # Ensure pad token is handled

    print("Original model parameters:")
    print_trainable_parameters(base_model)

    # Apply LoRA
    # For GPT2, common target_modules can be 'c_attn' (Query, Key, Value combined)
    # and sometimes 'c_proj' (output projection), 'c_fc' (feed-forward an_fc layer in MLP).
    # PEFT will try to find them.
    lora_model, lora_config = apply_lora(
        base_model,
        r=4, # Smaller rank for faster demo
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["c_attn"], # Specify for GPT2, can also try without for PEFT's auto-detection
        task_type=TaskType.CAUSAL_LM
    )

    print("\nLoRA model parameters (should show a small percentage as trainable):")
    # The print_trainable_parameters is already called inside apply_lora via peft_model.print_trainable_parameters()

    # --- Test with PyTorch Lightning ---
    print("\n--- Testing LoRA model with PyTorch Lightning ---")
    dummy_texts_train = [
        "LoRA fine-tuning example with GPT-2.",
        "Parameter-efficient tuning is great.",
        "Training only a few adapter weights."
    ] * 10
    dummy_texts_val = [
        "Validating the LoRA adapted model.",
        "Checking perplexity on validation set."
    ] * 5

    # Ensure tokenizer has pad_token_id set, especially if it was newly added
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        # If model vocab was resized, ensure lora_model is also aware or use the tokenizer that matches it
        lora_model.config.pad_token_id = tokenizer.pad_token_id


    train_dataloader = get_dataloader(dummy_texts_train, tokenizer, batch_size=2, max_length=64)
    val_dataloader = get_dataloader(dummy_texts_val, tokenizer, batch_size=2, max_length=64, shuffle=False)

    # Estimate total_training_steps for scheduler
    num_epochs = 1
    total_training_steps = len(train_dataloader) * num_epochs

    # Note: Only LoRA parameters should be optimized.
    # PEFT handles this by setting requires_grad correctly.
    # The BaseLLMSystem's optimizer will pick up only trainable parameters.
    lora_llm_system = BaseLLMSystem(lora_model, tokenizer, learning_rate=1e-4, total_training_steps=total_training_steps)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=True,
        enable_checkpointing=False,
        # fast_dev_run=True # for a quick test
    )

    print("Starting LoRA fine-tuning...")
    trainer.fit(lora_llm_system, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print("LoRA fine-tuning finished.")

    # Test generation after LoRA fine-tuning
    lora_model.eval() # Ensure model is in eval mode
    prompt = "Parameter-efficient fine-tuning is"
    inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)
    print(f"\nGenerating text with LoRA model from prompt: '{prompt}'")

    with torch.no_grad():
        generated_ids = lora_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=30, # Generate 30 new tokens
            pad_token_id=tokenizer.pad_token_id # Crucial for generation
        )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

    # To save LoRA adapters:
    # lora_model.save_pretrained("my_lora_adapters")
    # To load them later:
    # from peft import PeftModel
    # base_model = load_base_model(MODEL_NAME) # Load the original base model
    # lora_model_loaded = PeftModel.from_pretrained(base_model, "my_lora_adapters")