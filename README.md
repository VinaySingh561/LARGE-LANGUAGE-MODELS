🧠 GPT-2 From Scratch + Fine-Tuning on Harry Potter
This repository contains a full, modular implementation of the GPT-2 architecture from scratch using PyTorch and Google Colab. It includes all key components of a Transformer-based language model, a custom tokenizer, and fine-tuning on the Harry Potter book series.

🔧 Features
✅ Byte-Pair Encoding (BPE) tokenizer (LLM_tokenizer_from_scratch.ipynb, Byte_Pair_Encoding.ipynb)

✅ Token embeddings and positional encodings (Token_embeddings.ipynb, Positional_Embeddings.ipynb)

✅ Layer normalization and residual (shortcut) connections

✅ Multi-head attention and causal attention

✅ Custom training loop with PyTorch

✅ Fine-tuning on Harry Potter corpus using pretrained GPT-2 weights

📁 Notebooks Overview

File	Description
LLM_tokenizer_from_scratch.ipynb	Implements a custom BPE tokenizer
Byte_Pair_Encoding.ipynb	Byte Pair Encoding (BPE) logic
Token_embeddings.ipynb	Token and positional embeddings
Multi_Head_Attention.ipynb	Multi-head attention mechanism
Causal_Attention.ipynb	Implements causal masking for GPT-style models
Layer_Normalization_in_LLMs.ipynb	Layer norm module
Shortcut_connection.ipynb	Residual connections in transformers
Coding_the_entire_GPT_model.ipynb	Assembles full GPT model from components
Fine_tuning_the_LLM.ipynb	Fine-tunes the GPT model on the Harry Potter dataset
Inp_op_targets_using_PyTorch.ipynb	Prepares training inputs/targets
Simplified_Attention_Mechanism.ipynb	Minimal attention demo for understanding
gpt_download3.py	Download and setup script (e.g., datasets, weights)
📚 Dataset
The fine-tuning dataset is based on the Harry Potter book series.

Preprocessing includes tokenization via tiktoken and chunking into model-friendly sequences.

🚀 Results
After fine-tuning, the model is capable of generating stylistically consistent text in the voice of J.K. Rowling's Harry Potter universe. Sample outputs can be viewed in Fine_tuning_the_LLM.ipynb.

🛠️ Tech Stack
Python, PyTorch

Google Colab

tiktoken (for tokenizer compatibility)

Manual implementation of GPT-2 architecture (no Hugging Face / Transformers library used)

🧠 Learning Goals
This project was built to understand and implement transformer language models from the ground up, without relying on high-level libraries. It’s ideal for those looking to dive deep into:

GPT-2 architecture

Transformer internals

Tokenization and embeddings

Custom training workflows
