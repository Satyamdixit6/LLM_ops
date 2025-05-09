# llm_pipeline/base_llm_system.py

import torch
import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup

class BaseLLMSystem(pl.LightningModule):
    def __init__(self, model, tokenizer, learning_rate=5e-5, warmup_steps=0, total_training_steps=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps # Required for scheduler
        self.save_hyperparameters(ignore=['model', 'tokenizer']) # Saves lr, warmup_steps, etc.

        # For perplexity calculation
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer else -100)


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def common_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        # Shift logits and labels for perplexity if necessary (already handled by HuggingFace CausalLM loss)
        # For manual perplexity:
        # shift_logits = outputs.logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # active_loss = attention_mask[..., 1:].ne(0).view(-1)
        # active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
        # active_labels = shift_labels.view(-1)[active_loss]
        # loss_val = self.loss_fn(active_logits, active_labels)
        # perplexity = torch.exp(loss_val)

        perplexity = torch.exp(loss) # Approximation if using model's direct loss

        return loss, perplexity

    def training_step(self, batch, batch_idx):
        loss, perplexity = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, perplexity = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_perplexity", perplexity, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, perplexity = self.common_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_perplexity", perplexity, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        if self.total_training_steps is None:
            # Calculate total_training_steps if not provided
            # This requires knowledge of the DataLoader and number of epochs
            # For simplicity, we might require it to be passed or estimate it
            if self.trainer and self.trainer.datamodule:
                train_dataloader = self.trainer.datamodule.train_dataloader()
                self.total_training_steps = (
                    len(train_dataloader) // self.trainer.accumulate_grad_batches * self.trainer.max_epochs
                )
            elif self.trainer:
                 self.total_training_steps = self.trainer.estimated_stepping_batches
            else:
                # Fallback if trainer info is not available at this stage
                # User should provide this or it will be set later
                print("Warning: total_training_steps not set for scheduler. Scheduler might not work as expected.")
                return optimizer # Return optimizer without scheduler

        if self.total_training_steps is not None and self.total_training_steps > 0 :
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_training_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", # "epoch" or "step"
                    "frequency": 1
                }
            }
        else:
            return optimizer


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # For generation
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50, # Adjust as needed
            num_beams=5,   # Example beam search
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts


if __name__ == "__main__":
    from data_utils import get_tokenizer, get_dataloader
    from model_utils import load_base_model

    MODEL_NAME = "gpt2" # "distilgpt2" for faster testing
    tokenizer = get_tokenizer(MODEL_NAME)
    model_hf = load_base_model(MODEL_NAME)

    dummy_texts_train = [
        "This is a training sentence.", "Pytorch Lightning is cool for training.",
        "LLMs need data.", "Let's train a model."
    ] * 10 # More data for training
    dummy_texts_val = [
        "This is a validation sentence.", "Validating the LLM performance.",
        "How well does it generalize?", "Check validation loss."
    ] * 5

    train_dataloader = get_dataloader(dummy_texts_train, tokenizer, batch_size=2, max_length=64)
    val_dataloader = get_dataloader(dummy_texts_val, tokenizer, batch_size=2, max_length=64, shuffle=False)

    # Estimate total_training_steps
    num_epochs = 1
    total_training_steps = len(train_dataloader) * num_epochs

    llm_system = BaseLLMSystem(model_hf, tokenizer, total_training_steps=total_training_steps)

    # Minimal trainer for testing
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=True,
        enable_checkpointing=False,
        # fast_dev_run=True # Uncomment for a quick test run
    )

    print("Starting training...")
    trainer.fit(llm_system, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print("Training finished.")

    print("\nStarting validation...")
    trainer.validate(llm_system, dataloaders=val_dataloader)
    print("Validation finished.")

    # Test prediction
    print("\nTesting prediction...")
    test_texts = ["Translate this English text to French: Hello world.", "Summarize this document:"]
    # Note: tokenizer should be configured for tasks like translation/summarization
    # For simple generation from prompt:
    test_texts_prompts = ["Once upon a time, in a land far away,", "The future of AI is"]
    pred_dataloader = get_dataloader(test_texts_prompts, tokenizer, batch_size=1, max_length=32, shuffle=False) # Max length is for prompt
    predictions = trainer.predict(llm_system, dataloaders=pred_dataloader)
    for batch_preds in predictions:
        for text in batch_preds:
            print(f"Generated: {text}")