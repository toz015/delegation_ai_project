# 🚀 New Instance Setup Guide

## 📋 **What You Need**

1. **requirements.txt** - Python package list
2. **setup_environment.sh** - Automated setup script
3. **Your HuggingFace token**: `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`

## 🔧 **Step-by-Step Setup**

### **Step 1: Transfer Files to New Instance**
```bash
# Copy these files to your new instance:
# - requirements.txt
# - setup_environment.sh
# - NEW_INSTANCE_SETUP.md
```

### **Step 2: Run Setup Script**
```bash
# Make script executable
chmod +x setup_environment.sh

# Run the setup
./setup_environment.sh
```

### **Step 3: Restart Terminal**
```bash
source ~/.bashrc
```

### **Step 4: Set HuggingFace Token**
```bash
export HUGGING_FACE_HUB_TOKEN="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"
```

### **Step 5: Test Installation**
```bash
python3 -c "import vllm, torch, transformers; print('✅ All packages installed!')"
```

## 🎯 **Running Different Models**

### **Option 1: Different Model on Same Instance Type**
- Use different `model_ckpt` in your scripts
- Adjust `gpu_memory_utilization` based on model size
- Modify `max_model_len` as needed

### **Option 2: Different Instance Type**
- Use larger GPU instances for bigger models
- Adjust memory settings accordingly

## ⚠️ **Important Notes**

1. **Both instances can run simultaneously** - they're independent
2. **Each instance needs its own HuggingFace token** (or use the same one)
3. **GPU memory is per-instance** - no conflicts between instances
4. **Models are downloaded per-instance** (can share if needed)

## 🔍 **Troubleshooting**

- **CUDA issues**: Make sure NVIDIA drivers are installed
- **Memory issues**: Adjust `gpu_memory_utilization` in scripts
- **Package conflicts**: Use `pip3 install --user` for user-level installation

## 📞 **Need Help?**

Check the original instance for working configurations and copy them over!
