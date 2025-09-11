# Apple M4 Setup Instructions for TNC

## 1. Install Dependencies

```bash
# Install Miniforge (conda for Apple Silicon)
brew install miniforge

# Create environment
conda create -n tnc python=3.10
conda activate tnc

# Install PyTorch with Apple Silicon support
conda install pytorch torchvision torchaudio -c pytorch

# Install other ML packages
conda install numpy pandas matplotlib seaborn scikit-learn

# Install additional packages
pip install tslearn TimeSynth statsmodels wfdb pyod

# Optional: Jupyter for experimentation
conda install jupyter
```

## 2. Enable Apple Silicon Acceleration

Create a device detection script that uses MPS (Metal Performance Shaders) for faster training:

```python
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")
```

## 3. Expected Performance on M4

- **CPU Training**: Works perfectly, good for small datasets
- **MPS Training**: 2-3x faster than CPU for larger models
- **Memory**: 8GB unified memory is shared between CPU/GPU
- **Speed**: M4 is very fast for ML workloads, comparable to entry-level GPUs

## 4. Quick Test

```bash
# Test if everything works
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## 5. Training Recommendations

For your M4 MacBook Air:

- **Small datasets (simulation)**: Runs very fast
- **Medium datasets (HAR)**: Good performance 
- **Large datasets (ECG)**: May need to reduce batch size if memory issues

## 6. Memory Management

If you run into memory issues:
```python
# Add this to training scripts
if device.type == 'mps':
    torch.mps.empty_cache()  # Clear MPS cache
```

## 7. Compatibility Notes

✅ All TNC code works on Apple Silicon
✅ PyTorch operations are optimized for M4
✅ Automatic fallback to CPU if needed
✅ No code changes required (already fixed device detection)

The M4 chip is excellent for machine learning - you'll have a great experience!
