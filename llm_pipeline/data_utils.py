# llm_pipeline/data_utils.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class CustomTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # For language modeling, input_ids are labels too (shifted in the model)
        return {
            "input_ids": encoding.input_ids.squeeze(),
            "attention_mask": encoding.attention_mask.squeeze(),
            "labels": encoding.input_ids.squeeze().clone() # Ensure labels are separate
        }

def get_tokenizer(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Common practice for GPT-like models
    return tokenizer

def get_dataloader(texts, tokenizer, batch_size=4, max_length=512, shuffle=True):
    dataset = CustomTextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Example usage (you can test this file independently)
if __name__ == "__main__":
    dummy_texts = [
        "This is the first sentence for our LLM.",
        "Pytorch Lightning makes training easier.",
        "Large Language Models are powerful.",
        "We are building a modular pipeline.",
        "Tokenization is a crucial first step.",
        "Let's test the data loader functionality.",
        "Another example sentence to increase dataset size.",
        "The final text for this small dummy dataset."
    ]
    tokenizer = get_tokenizer()
    dataloader = get_dataloader(dummy_texts, tokenizer, batch_size=2)

    for batch in dataloader:
        print("Batch input_ids shape:", batch["input_ids"].shape)
        print("Batch attention_mask shape:", batch["attention_mask"].shape)
        print("Batch labels shape:", batch["labels"].shape)
        print("-" * 20)
        break