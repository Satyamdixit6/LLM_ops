# llm_pipeline/adaptations/rag.py

import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration # For a more integrated RAG
from typing import List, Dict, Union
import numpy as np

# --- Simplified Custom RAG Components ---

class SimpleDocumentStore:
    def __init__(self, documents: List[str], embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.documents = documents
        self.embedder = SentenceTransformer(embedding_model_name)
        self.embeddings = self._embed_documents()
        self.index = self._build_faiss_index()

    def _embed_documents(self):
        print(f"Embedding {len(self.documents)} documents...")
        return self.embedder.encode(self.documents, convert_to_tensor=False, show_progress_bar=True)

    def _build_faiss_index(self):
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Using L2 distance
        index.add(np.array(self.embeddings, dtype=np.float32))
        print(f"FAISS index built with {index.ntotal} documents.")
        return index

    def retrieve_documents(self, query: str, k: int = 3) -> List[Dict[str, Union[str, float]]]:
        query_embedding = self.embedder.encode([query], convert_to_tensor=False)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)

        distances, indices = self.index.search(query_embedding_np, k)

        results = []
        for i in range(len(indices[0])):
            doc_idx = indices[0][i]
            dist = distances[0][i]
            results.append({
                "text": self.documents[doc_idx],
                "score": float(dist) # Lower L2 distance is better
            })
        return results

def augment_prompt_with_context(prompt: str, retrieved_docs: List[Dict[str, Union[str, float]]]) -> str:
    context_str = "\n\nRelevant Context:\n"
    for i, doc in enumerate(retrieved_docs):
        context_str += f"[{i+1}] {doc['text']}\n"
    
    augmented_prompt = f"{context_str}\nQuestion: {prompt}\nAnswer:"
    return augmented_prompt


# --- PyTorch Lightning System adapted for RAG (Conceptual) ---
# For a true RAG model like Hugging Face's RagSequenceForGeneration,
# the architecture itself handles retrieval and generation jointly.
# Our simplified approach here is to augment the prompt for a standard LLM.

class RAGLLMSystem(torch.nn.Module): # Not a LightningModule for this simple wrapper
    def __init__(self, llm_model, llm_tokenizer, document_store: SimpleDocumentStore, k_retrieval=3):
        super().__init__()
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.document_store = document_store
        self.k_retrieval = k_retrieval
        self.device = next(self.llm_model.parameters()).device


    def generate(self, prompt: str, max_new_tokens=50, **generate_kwargs):
        retrieved_docs = self.document_store.retrieve_documents(prompt, k=self.k_retrieval)
        augmented_prompt = augment_prompt_with_context(prompt, retrieved_docs)
        
        print(f"\n--- RAG ---")
        print(f"Original Prompt: {prompt}")
        print(f"Retrieved Docs: {[doc['text'][:100] + '...' for doc in retrieved_docs]}") # Print snippets
        print(f"Augmented Prompt (first 200 chars): {augmented_prompt[:200]}...")
        print("--- End RAG ---")

        inputs = self.llm_tokenizer(augmented_prompt, return_tensors="pt", truncation=True, max_length=self.llm_tokenizer.model_max_length - max_new_tokens).to(self.device)
        
        with torch.no_grad():
            output_ids = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                **generate_kwargs
            )
        
        generated_text = self.llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Often, the answer part needs to be extracted from the full generated text
        # which includes the augmented prompt.
        # A simple way is to find "Answer:" and take text after it.
        answer_marker = "Answer:"
        answer_start_index = generated_text.rfind(answer_marker)
        if answer_start_index != -1:
            answer = generated_text[answer_start_index + len(answer_marker):].strip()
        else:
            # Fallback if "Answer:" marker is not in the output (e.g., model didn't follow instruction)
            # This depends on how the prompt was structured and what the LLM generates.
            # We might just return the part after the original augmented prompt.
            input_tokens_len = inputs.input_ids.shape[1]
            answer = self.llm_tokenizer.decode(output_ids[0][input_tokens_len:], skip_special_tokens=True).strip()


        return answer, retrieved_docs, augmented_prompt


if __name__ == "__main__":
    from ..model_utils import load_base_model # Relative import
    from ..data_utils import get_tokenizer

    # 1. Setup Document Store
    sample_documents = [
        "The Eiffel Tower is located in Paris, France and was completed in 1889.",
        "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab (FAIR).",
        "The capital of Japan is Tokyo, a bustling metropolis known for its blend of modern and traditional culture.",
        "Large Language Models (LLMs) like GPT-3 are trained on vast amounts of text data.",
        "The Amazon rainforest is the world's largest tropical rainforest, famed for its biodiversity.",
        "The first person to walk on the Moon was Neil Armstrong in 1969 during the Apollo 11 mission.",
        "Photosynthesis is the process used by plants, algae, and some bacteria to convert light energy into chemical energy.",
        "Mount Everest is the Earth's highest mountain above sea level, located in the Himalayas."
    ]
    doc_store = SimpleDocumentStore(documents=sample_documents)

    # Test retrieval
    query = "What is PyTorch?"
    retrieved = doc_store.retrieve_documents(query, k=2)
    print(f"\nQuery: {query}")
    for doc in retrieved:
        print(f"  - Retrieved: {doc['text']} (Score: {doc['score']:.4f})")

    # 2. Load a base LLM (e.g., GPT-2) for generation
    MODEL_NAME = "gpt2" # "distilgpt2" for faster testing
    llm_tokenizer = get_tokenizer(MODEL_NAME)
    # Ensure pad token is set for generation
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
    llm_model_hf = load_base_model(MODEL_NAME, add_pad_token=True)
    llm_model_hf.config.pad_token_id = llm_tokenizer.pad_token_id # Ensure model config has it too
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model_hf.to(device)
    llm_model_hf.eval()


    # 3. Create RAG-LLM System
    rag_system = RAGLLMSystem(
        llm_model=llm_model_hf,
        llm_tokenizer=llm_tokenizer,
        document_store=doc_store,
        k_retrieval=2
    )

    # 4. Test RAG generation
    prompts = [
        "Where is the Eiffel Tower?",
        "Who developed PyTorch?",
        "What is the highest mountain on Earth?",
        "Tell me about LLMs."
    ]

    for p in prompts:
        print(f"\nProcessing prompt: {p}")
        answer, _, aug_prompt = rag_system.generate(p, max_new_tokens=60, num_beams=3) # Added num_beams for potentially better generation
        print(f"Generated Answer: {answer}")

    # Example with a question where context might be less direct
    # (to see if the LLM can synthesize or if it just relies on context)
    no_context_prompt = "What is the color of the sky on Mars?" # Not in our docs
    print(f"\nProcessing prompt (likely no direct context): {no_context_prompt}")
    answer, retrieved, _ = rag_system.generate(no_context_prompt, max_new_tokens=30)
    print(f"Retrieved for Mars sky: {[doc['text'][:50] for doc in retrieved]}")
    print(f"Generated Answer: {answer}")
    print("-" * 30)