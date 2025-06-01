
## This project is a modular and optimized Large Language Model (LLM) pipeline built with PyTorch and PyTorch Lightning, focusing on efficient data handling, model training, fine-tuning, and inference. It incorporates techniques like LoRA, quantization, Retrieval Augmented Generation (RAG), FlashAttention, and custom CUDA kernels for performance optimization.



Key Features
Efficient Data Handling: Custom data loaders and tokenization for text datasets.

Model Optimization: Supports mixed-precision training, quantization (dynamic/static), LoRA, FlashAttention, and custom CUDA kernels.

Modular Design: Organized modules for easy adaptation and experimentation.

Performance Goals: Aims to reduce inference latency (~30%) and model size (25-50%) while maintaining accuracy.

Experiment Management: Run experiments via main.py with command-line arguments.

Project Structure


Setup
Prerequisites:
Python 3.8+

PyTorch (CUDA-compatible if using GPU)

NVIDIA GPU, CUDA Toolkit, nvcc, C++ compiler

Install dependencies: pip install -r requirements.txt

Build CUDA kernel: cd llm_pipeline/adaptations/custom_cuda_kernel && python setup.py build_ext --inplace

Key Dependencies:
torch, transformers, pytorch-lightning, peft, sentence-transformers, faiss-cpu (or faiss-gpu), accelerate

Optional: flash-attn (GPU-specific)

How to Run
Run experiments with python main.py --adaptation <name> [options]:
Baseline: python main.py --adaptation baseline --model_name distilgpt2

Custom CUDA Kernel: python main.py --adaptation custom_cuda_kernel --model_name distilgpt2

Quantization: python main.py --adaptation quantization --quant_type dynamic --model_name distilgpt2

LoRA: python main.py --adaptation lora --model_name distilgpt2 --lora_r 8 --lora_alpha 16

RAG: python main.py --adaptation rag --model_name gpt2

FlashAttention: python main.py --adaptation flash_attention --flash_attention_model_name gpt2 --trust_remote_code

Outputs are saved in llm_pipeline_output/.
Future Improvements
Enhance RAG with advanced strategies.

Add Quantization Aware Training (QAT).

Support distributed training.

Include Dockerization and comprehensive benchmarking.

