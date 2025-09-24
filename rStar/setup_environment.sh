#!/bin/bash

# Environment Setup Script for delegation_ai_project
# Run this on the new instance to set up the same environment

echo "🚀 Setting up delegation_ai_project environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.10 if not present
if ! command -v python3.10 &> /dev/null; then
    echo "🐍 Installing Python 3.10..."
    sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip
fi

# Install system dependencies for vLLM and PyTorch
echo "🔧 Installing system dependencies..."
sudo apt install -y build-essential git curl wget

# Install CUDA dependencies (if needed)
echo "🎮 Installing CUDA dependencies..."
sudo apt install -y nvidia-cuda-toolkit

# Create user-level pip directory
mkdir -p ~/.local/lib/python3.10/site-packages

# Install Python packages from requirements.txt
echo "📚 Installing Python packages..."
pip3 install --user -r requirements.txt

# Set up environment variables
echo "🔑 Setting up environment variables..."
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"' >> ~/.bashrc

# Install HuggingFace CLI
echo "🤗 Installing HuggingFace CLI..."
pip3 install --user huggingface_hub

echo "✅ Environment setup complete!"
echo "🔄 Please restart your terminal or run: source ~/.bashrc"
echo "🔑 Don't forget to set your HuggingFace token: export HUGGING_FACE_HUB_TOKEN='your_token_here'"
