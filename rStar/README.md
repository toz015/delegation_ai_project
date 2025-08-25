# rStar Multi-Model Implementation

This repository contains a clean implementation of rStar with support for multiple language models and datasets.

## 🚀 **Models Supported**

- **Qwen2-7B-Instruct** - For GSM8K generation
- **Llama3.1-8B-Instruct** - For GSM8K generation  
- **Mistral-7B-Instruct-v0.2** - For GSM8K and STG generation

## 📁 **Configuration Files**

### **GSM8K Models:**
- `demo_rstar_gsm8k_qwen.py` - Qwen2-7B configuration
- `demo_rstar_gsm8k_llama3.py` - Llama3.1-8B configuration
- `demo_rstar_gsm8k_mistral.py` - Mistral-7B configuration

### **STG Dataset:**
- `demo_rstar_stg_mistral.py` - STG dataset with Mistral-7B

### **Delegation Algorithms:**
- `delegation_algorithm.py` - Basic delegation implementation
- `delegation_algorithm_multiple_choice.py` - Multiple choice discriminator
- `delegation_algorithm_qwen.py` - Qwen-specific delegation

## 🔧 **Key Features**

- **Multiple Model Support** - Easy switching between different LLMs
- **STG Dataset Support** - True/False question dataset
- **Delegation Algorithms** - Collaborative learning between models
- **Memory Optimized** - Configurable GPU memory usage
- **Flexible Reasoning** - A1/A5 path configuration

## 🚀 **Quick Start**

1. **Clone the repository:**
   ```bash
   git clone <your-new-repo-url>
   cd rstar-clean
   ```

2. **Run any model configuration:**
   ```bash
   python3 demo_rstar_stg_mistral.py      # STG with Mistral
   python3 demo_rstar_gsm8k_qwen.py      # GSM8K with Qwen
   python3 demo_rstar_gsm8k_llama3.py    # GSM8K with Llama3
   python3 demo_rstar_gsm8k_mistral.py   # GSM8K with Mistral
   ```

## 📊 **Model Configurations**

| Model | Dataset | A5 (Rephrasing) | Rollouts | Notes |
|-------|---------|------------------|----------|-------|
| Qwen2-7B | GSM8K | Disabled | 16 | Fast generation |
| Llama3.1-8B | GSM8K | Enabled | 16 | Enhanced reasoning |
| Mistral-7B | GSM8K | Enabled | 24 | Balanced approach |
| Mistral-7B | STG | Enabled | 24 | True/False questions |

## 🎯 **Use Cases**

- **Research** - Compare different model performances
- **Education** - Study reasoning patterns across models
- **Development** - Build on top of the delegation framework
- **Evaluation** - Assess model capabilities on different datasets

## 📝 **Requirements**

- Python 3.8+
- vLLM for model inference
- HuggingFace Transformers
- CUDA-compatible GPU
- Sufficient GPU memory (8GB+ recommended)

## 🔗 **Original Repository**

This is a clean, focused version of the original rStar implementation, containing only the essential model configurations and algorithms.
