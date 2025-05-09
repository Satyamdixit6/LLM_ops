# llm_pipeline/model_utils.py

from transformers import AutoModelForCausalLM, AutoConfig

def load_base_model(model_name="gpt2", add_pad_token=True):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

    if add_pad_token:
        # Resize token embeddings if pad token was added to tokenizer
        # This is important if the tokenizer's vocab size changed
        from data_utils import get_tokenizer # Avoid circular import if not needed for this step
        tokenizer = get_tokenizer(model_name)
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= model.config.vocab_size:
             model.resize_token_embeddings(len(tokenizer))
        elif tokenizer.pad_token is None and model.config.pad_token_id is None:
            # If model doesn't have a pad token, and tokenizer didn't either but we assigned eos_token
            # Ensure consistency if a new pad token was effectively added
            if tokenizer.eos_token_id is not None: # GPT2 typically has eos_token
                 # If we assigned tokenizer.pad_token = tokenizer.eos_token,
                 # and the model already knows eos_token, no resize might be needed
                 # unless the vocab size itself is an issue.
                 # This step is subtle and depends on specific model/tokenizer state.
                 # For robustness with newly added pad tokens:
                 model.config.pad_token_id = tokenizer.pad_token_id # Ensure model config is aware

    return model

if __name__ == "__main__":
    model = load_base_model()
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Model config: {model.config}")
    # Test with tokenizer to ensure vocab sizes match after potential resizing
    from data_utils import get_tokenizer
    tokenizer = get_tokenizer()
    if model.config.vocab_size != len(tokenizer):
        print(f"Warning: Model vocab size ({model.config.vocab_size}) and tokenizer vocab size ({len(tokenizer)}) may mismatch after potential resize.")
    else:
        print("Model and tokenizer vocab sizes are consistent.")