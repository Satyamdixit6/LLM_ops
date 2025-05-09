# Modular and Optimized Large Language Model (LLM) Pipeline

This project demonstrates a modular and optimized Large Language Model (LLM) pipeline built with PyTorch and PyTorch Lightning. It showcases various techniques for efficient data handling, model training, fine-tuning, and inference, including parameter-efficient fine-tuning (LoRA), quantization, Retrieval Augmented Generation (RAG), FlashAttention integration, and custom CUDA kernel usage.

## Overview

The primary goal of this project is to architect and implement a flexible LLM pipeline that allows for easy integration and benchmarking of different optimization and adaptation strategies. This serves as a foundation for developing and deploying LLMs for various tasks while managing computational resources effectively.

**Key Project Contributions & Goals (Inspired by typical LLM optimization projects):**
* **Efficient Data Handling:** Engineered custom data loaders and advanced tokenization strategies in PyTorch to efficiently process and batch text datasets.
* **Model Optimization Benchmarking:** Implemented and provided a framework to benchmark model optimization techniques, including:
    * Mixed-precision training (supported by PyTorch Lightning).
    * Post-training static and dynamic quantization.
    * Low-Rank Adaptation (LoRA) for efficient fine-tuning.
    * FlashAttention for faster and memory-efficient attention mechanisms.
    * Integration of custom CUDA kernels for specialized operations.
* **Reproducibility & Scalability:** Designed for reproducibility and to facilitate scalable deployment workflows (conceptualized, Dockerization would be a next step).
* **Performance Tracking:** Utilized basic logging for tracking experimental parameters and performance metrics.

**Target Quantifiable Outcomes (Illustrative goals for such a pipeline):**
* Achieve significant reduction in average inference latency (e.g., targeting ~30% from a baseline) while preserving model accuracy (e.g., perplexity) within a small margin (e.g., 2%) of the full-precision baseline.
* Reduce model size (e.g., by approximately 25-50%) through techniques like quantization, enabling deployment on resource-constrained environments.

## Features Implemented

* **Baseline LLM System:** A core PyTorch Lightning module for standard training and evaluation of LLMs (e.g., GPT-2).
* **Custom CUDA Kernel Integration:** Example of integrating a simple custom CUDA kernel (Clamped ReLU) into the PyTorch model.
* **Quantization:**
    * Post-Training Dynamic Quantization.
    * Post-Training Static Quantization (with caveats for complex transformer models).
* **LoRA (Low-Rank Adaptation):** Efficient fine-tuning using the `peft` library.
* **RAG (Retrieval Augmented Generation):** A simplified RAG system for augmenting LLM prompts with retrieved documents from a FAISS-indexed document store.
* **FlashAttention Integration:** Demonstrates how to leverage FlashAttention 2 (via `attn_implementation` in Hugging Face `transformers`) for optimized attention.
* **Modular Design:** Adaptations are organized into separate modules for clarity and ease of use.
* **Experiment Management:** Central `main.py` script to run different pipeline configurations with command-line arguments.

## Project Structure

llm_pipeline/
├── main.py                     # Main script to run experiments and select adaptations
├── base_llm_system.py          # Core PyTorch Lightning LLM system
├── data_utils.py               # Data loading and preprocessing utilities
├── model_utils.py              # Utilities for loading Hugging Face models
│
├── adaptations/                # Modules for different LLM adaptations
│   ├── init.py
│   ├── quantization.py         # Quantization logic (static and dynamic)
│   ├── lora.py                 # LoRA implementation/integration using PEFT
│   ├── rag.py                  # RAG components (retriever, document store, augmented prompting)
│   ├── flash_attention.py      # FlashAttention integration utilities
│   └── custom_cuda_kernel/
│       ├── init.py
│       ├── kernel_wrapper.py   # PyTorch module wrapping the CUDA kernel
│       ├── custom_kernel.cu    # The custom CUDA kernel code (Clamped ReLU)
│       └── setup.py            # Script to build the CUDA kernel
│
├── configs/                    # Configuration files (e.g., for model, training)
│   └── base_config.yaml        # Conceptual YAML configuration
│
└── llm_pipeline_output/        # Default output directory for logs, saved models, etc. (created at runtime)
└── README.md                   # This file


## Setup and Installation

### Prerequisites
* Python 3.8+
* PyTorch (version compatible with your CUDA version if using GPU, e.g., PyTorch 1.13+ or 2.x)
* NVIDIA GPU with CUDA (recommended for most features, especially FlashAttention and custom CUDA kernel)
    * CUDA Toolkit (e.g., 11.8 or 12.1 - ensure compatibility with PyTorch and FlashAttention)
    * `nvcc` compiler for building the custom CUDA kernel.
* C++ compiler (e.g., g++) for building PyTorch extensions.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    A `requirements.txt` file is provided.
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not present, create it with the following content or install packages manually:
    ```txt
    # requirements.txt
    torch>=1.13.0 # Or newer, ensure CUDA compatibility if using GPU (e.g., torch==2.2.0+cu118)
    torchvision
    torchaudio
    transformers>=4.31.0 # For attn_implementation
    pytorch-lightning>=2.0.0
    tensorboard
    sentence-transformers
    faiss-cpu # Or faiss-gpu if you have a compatible environment
    peft>=0.7.0
    accelerate>=0.25.0
    # Only if you have a compatible GPU & CUDA for FlashAttention:
    # flash-attn --no-build-isolation # Install separately as per official instructions
    ```
    **Note on PyTorch:** Install the version of PyTorch that matches your CUDA version. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for specific commands.
    **Note on FlashAttention:** `flash-attn` installation can be tricky and is hardware/CUDA specific. Follow official [flash-attn installation guide](https://github.com/Dao-AILab/flash-attention#installation-and-features). It's listed as optional.

4.  **Build the Custom CUDA Kernel:**
    Navigate to the `custom_cuda_kernel` directory and run the build script:
    ```bash
    cd llm_pipeline/adaptations/custom_cuda_kernel/
    python setup.py build_ext --inplace
    cd ../../..  # Navigate back to the project's root directory (e.g., llm_pipeline/ or its parent)
    ```
    This will compile `custom_kernel.cu` and create a shared object file (`.so` or `.pyd`) in the same directory.

## How to Run

The main script `llm_pipeline/main.py` is used to run the pipeline with different adaptations.
Ensure you are in the directory containing `main.py` or can invoke it correctly (e.g., `python llm_pipeline/main.py ...`).

**General Command Structure:**
```bash
python main.py --adaptation <adaptation_name> [options]
Examples:
Baseline Training & Evaluation:
Uses distilgpt2 for faster local testing by default in main.py's arguments.

Bash

python main.py --adaptation baseline --model_name distilgpt2
Custom CUDA Kernel Integration:
(Assumes the kernel is built successfully)

Bash

python main.py --adaptation custom_cuda_kernel --model_name distilgpt2
Quantization (Dynamic):

Bash

python main.py --adaptation quantization --quant_type dynamic --model_name distilgpt2
Quantization (Static):
Note: Full static quantization of Hugging Face transformers using basic PyTorch API can be challenging. This is for demonstration.

Bash

python main.py --adaptation quantization --quant_type static --model_name distilgpt2
LoRA Fine-tuning:

Bash

python main.py --adaptation lora --model_name distilgpt2 --lora_r 8 --lora_alpha 16
RAG (Inference Demo):

Bash

python main.py --adaptation rag --model_name gpt2
FlashAttention (Inference Demo):
*Requires a compatible GPU, CUDA, transformers>=4.31, and flash-attn installed.
The --flash_attention_model_name argument can specify models known for good FA support.
--trust_remote_code might be needed for some models.

Bash

python main.py --adaptation flash_attention --flash_attention_model_name EleutherAI/gpt-neo-125M --trust_remote_code
To try with gpt2 (which might use PyTorch's SDPA as a fallback if direct FlashAttention-2 is not supported by the model's architecture in transformers):

Bash

python main.py --adaptation flash_attention --flash_attention_model_name gpt2 --trust_remote_code
Arguments:

--adaptation: Choose from baseline, custom_cuda_kernel, quantization, lora, rag, flash_attention.
--model_name: Base Hugging Face model (e.g., gpt2, distilgpt2).
--quant_type: dynamic or static (for quantization adaptation).
--lora_r, --lora_alpha: Parameters for LoRA.
--flash_attention_model_name: Specific model for FlashAttention testing.
--trust_remote_code: For Hugging Face from_pretrained.
Logs and any generated outputs (like LoRA adapters) will be saved in the llm_pipeline_output/ directory by default.

Modules Overview
main.py: Entry point for the pipeline. Parses arguments, loads components, and orchestrates the selected adaptation workflow.
base_llm_system.py: Defines BaseLLMSystem, a PyTorch Lightning LightningModule for handling standard LLM training, validation, testing, and prediction logic (loss calculation, optimization, perplexity).
data_utils.py: Contains CustomTextDataset for creating PyTorch datasets from text lists, get_tokenizer for loading Hugging Face tokenizers, and get_dataloader for creating DataLoader instances.
model_utils.py: Includes load_base_model for loading pre-trained Hugging Face models.
adaptations/:
custom_cuda_kernel/: Contains the .cu CUDA C++ source for a custom activation function, setup.py for compilation, and kernel_wrapper.py to expose it as a PyTorch nn.Module.
quantization.py: Implements quantize_model_dynamic and quantize_model_static for applying post-training quantization.
lora.py: Provides apply_lora function to modify a model with LoRA layers using the peft library.
rag.py: Implements SimpleDocumentStore (using sentence-transformers and faiss) and RAGLLMSystem for a retrieve-then-read RAG pipeline.
flash_attention.py: Includes try_load_model_with_flash_attention_2 to attempt loading models with FlashAttention 2 or SDPA via Hugging Face transformers.
Configuration
A conceptual configuration structure is provided in llm_pipeline/configs/base_config.yaml. While main.py primarily uses command-line arguments for simplicity in this example, a more extensive application would benefit from loading configurations from such YAML files using libraries like Hydra or OmegaConf.

Future Work & Improvements
Advanced RAG: Implement more sophisticated RAG strategies (e.g., joint training, better context management).
Quantization Aware Training (QAT): Add support for QAT for potentially better accuracy with quantized models.
Distributed Training: Expand examples for multi-GPU/multi-node training using PyTorch Lightning's capabilities.
Dockerization: Package the pipeline with Docker for easier deployment and reproducibility.
Comprehensive Benchmarking: Add more rigorous benchmarking for latency, throughput, and accuracy across adaptations.
Expanded Model Support: Test and ensure compatibility with a wider range of LLM architectures.
Configuration Management: Fully integrate a configuration library like Hydra.
Unit and Integration Tests: Implement a robust testing suite.
Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

License
This project is open-sourced under the MIT License. See the LICENSE file for more details
