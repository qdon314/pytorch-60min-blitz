# PyTorch 60-Minute Blitz — Notes & Implementations

This repository contains my personal walkthrough of the **PyTorch 60-Minute Blitz**, including annotated code, explanations, experiments, and small extensions beyond the official tutorial. It serves both as a reference for myself and a quick demonstration of practical PyTorch fundamentals.

## 📘 Contents

### 1. Tensors
- Creating and manipulating tensors  
- Converting between NumPy arrays and PyTorch tensors  
- GPU (CUDA/MPS) acceleration basics  

### 2. Autograd
- Automatic differentiation mechanics  
- Computing gradients  
- Tracking computation graphs  

### 3. Neural Networks (`torch.nn`)
- Building simple feed-forward networks  
- Understanding layers, activations, and forward passes  
- Inspecting model parameters  

### 4. Training a Classifier
- Loading datasets with `torchvision`  
- Data augmentation and normalization  
- Training loop (forward → loss → backward → optimizer)  
- Evaluating accuracy  
- Saving and loading models  

## 🚀 Goals of This Repo

- Build intuition around PyTorch’s core concepts  
- Create a clean reference I can revisit  
- Prepare for deeper ML / DL work (fast.ai, Hugging Face courses, etc.)  
- Demonstrate foundational ML engineering skills in a real codebase  

## 📂 Structure
```bash
/notebooks/        # Annotated Jupyter notebooks for each section
/src/              # Standalone Python scripts for reproducible runs
/data/             # Downloaded datasets (gitignored)
README.md
```

## 🧰 Requirements

- Python 3.10+  
- PyTorch (CPU or GPU)  
- torchvision  
- Jupyter Notebook or VS Code  

Install dependencies:

```bash
pip install torch torchvision matplotlib jupyter
```

```bash
# E.g.
python src/classifier.py
```
