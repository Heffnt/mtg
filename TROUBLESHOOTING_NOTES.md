# PyTorch/Unsloth Environment Setup - Lessons Learned

**Date:** October 19, 2025
**Issue:** Setting up Unsloth for LLM fine-tuning with proper PyTorch compatibility

---

## Key Lessons

### 1. **Unsloth Has Strict PyTorch Version Requirements**

- **Unsloth only supports PyTorch up to 2.5.x** (as of Oct 2025)
- PyTorch 2.6+ is NOT supported - will cause errors
- PyTorch 2.8 is definitely incompatible
- Always check Unsloth compatibility before upgrading PyTorch

**Sources:**
- GitHub issue: https://github.com/unslothai/unsloth/issues/1825
- Unsloth docs: https://docs.unsloth.ai/get-started/installing-+-updating/pip-install

### 2. **torchvision Version Mismatch Causes Cryptic Errors**

**Symptom:**
```
RuntimeError: operator torchvision::nms does not exist
```

**Root Cause:**
- PyTorch and torchvision versions must be exactly matched
- Installing from different sources (pip vs conda vs different indexes) can cause binary incompatibility
- Even if versions look correct (e.g., both 2.5.1), binaries may not match

**Solution:**
- Always install PyTorch, torchvision, and torchaudio together from the same source
- Use `--index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1 builds
- Use `uv pip install` for speed, but ensure consistent sources

### 3. **torchao Version Must Match PyTorch Version**

**Symptom:**
```
AttributeError: module 'torch' has no attribute 'int1'
```

**Root Cause:**
- torchao 0.13.0+ requires PyTorch 2.6+ for `torch.int1` dtype
- Newer versions of transformers/accelerate can pull in incompatible torchao versions

**Solution:**
- Pin torchao version explicitly: `torchao==0.12.0` for PyTorch 2.5.1
- Compatibility matrix:
  - torchao 0.12.0 → PyTorch 2.5.0, 2.6.0, 2.7.1
  - torchao 0.13.0 → PyTorch 2.6.0, 2.7.1, 2.8.0
  - torchao 0.14.x → PyTorch 2.9.0 (nightly)

**Source:** https://github.com/pytorch/ao/issues/2919

### 4. **Dependency Resolution Can Upgrade PyTorch Without Warning**

**Problem:**
- Installing packages like `unsloth` or `transformers` can trigger PyTorch upgrades
- Pip's dependency resolver doesn't always respect existing package versions
- This breaks the carefully constructed PyTorch 2.5.1 environment

**Solution:**
- Use `--no-deps` when installing packages that might upgrade PyTorch
- Install dependencies manually in a controlled manner
- Pin PyTorch version in requirements/scripts

### 5. **Environment Isolation is Critical**

**Issue Encountered:**
- Packages were installing to base conda environment instead of `mtgenv`
- `conda activate` doesn't work in bash scripts without proper shell initialization

**Solution:**
- Always verify `which python` points to the correct environment
- For scripts: Use absolute paths or ensure conda is initialized
- Consider using `uv` within activated environments for speed

### 6. **Model Tokenizer Compatibility Issues**

**Symptom:**
```
RuntimeError: Unsloth: The tokenizer does not have a
{% if add_generation_prompt %} for generation purposes.
```

**Root Cause:**
- Some models (like `qihoo360/TinyR1-32B`) have incomplete/incompatible chat templates
- Unsloth requires specific tokenizer features for training

**Solution:**
- Use well-supported models with proper chat templates
- DeepSeek-R1-Distill-Qwen-32B is a good alternative (based on Qwen2.5)
- Check model card for chat template support before starting training

---

## Working Configuration (Oct 2025)

### Python & Core
- Python: 3.12
- CUDA: 12.1 (not 12.8!)

### PyTorch Stack
```bash
torch==2.5.1  # From pytorch.org/whl/cu121
torchvision==0.20.1
torchaudio==2.5.1
torchao==0.12.0  # CRITICAL: Pin this version!
```

### ML Frameworks
```bash
transformers  # Latest compatible
datasets
accelerate
peft
trl
wandb
deepspeed
bitsandbytes
einops
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
unsloth_zoo
```

### Installation Command Sequence
```bash
# 1. Create minimal conda env
conda env create -f environment.yml  # Just Python + pip

# 2. Install uv for speed
pip install uv

# 3. Install PyTorch stack (EXACT versions, SAME source)
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Pin torchao
uv pip install torchao==0.12.0

# 5. Install ML frameworks (use --no-deps to prevent PyTorch upgrade)
uv pip install --no-deps transformers datasets accelerate peft trl \
    wandb bitsandbytes einops

# 6. Install dependencies manually
uv pip install huggingface_hub safetensors pyyaml regex requests \
    packaging numpy filelock fsspec tokenizers psutil dill \
    multiprocess pyarrow pandas xxhash aiohttp

# 7. Install Unsloth (with --no-deps to prevent upgrades)
uv pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
uv pip install --no-deps unsloth_zoo

# 8. Install Unsloth dependencies
uv pip install tyro cut_cross_entropy mistral_common hf_transfer
```

---

## Recommended Models for Unsloth

### ✅ Known Working Models
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` - Reasoning-focused, MIT license
- Qwen2.5 series - Generally well-supported
- Llama 3 series - Well-supported

### ❌ Models with Known Issues
- `qihoo360/TinyR1-32B` - Tokenizer chat template issues

---

## Debugging Tips

### 1. Verify PyTorch Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
```

### 2. Test Unsloth Loading
```bash
python -c 'from unsloth import FastLanguageModel; print("✓ Unsloth loaded")'
```

### 3. Check Package Versions
```bash
pip list | grep -E "(torch|unsloth|transformers|torchao)"
```

### 4. Verify Environment
```bash
which python  # Should point to conda env, not base
echo $CONDA_DEFAULT_ENV  # Should show 'mtgenv'
```

---

## Common Errors and Quick Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `operator torchvision::nms does not exist` | PyTorch/torchvision version mismatch | Reinstall both from same source |
| `module 'torch' has no attribute 'int1'` | torchao version too new | Downgrade to `torchao==0.12.0` |
| `Torch = X.X.X too new!` | PyTorch version unsupported | Downgrade to PyTorch 2.5.1 |
| Tokenizer `add_generation_prompt` error | Model incompatibility | Switch to Qwen2.5 or DeepSeek model |
| Packages installing to wrong env | Conda env not activated | Check `which python` |

---

## Tools & Resources

### Package Managers
- **uv**: 10-100x faster than pip, use for all installations
  - Install: `pip install uv`
  - Usage: `uv pip install <package>`

### Version Compatibility Checkers
- PyTorch compatibility: https://pytorch.org/get-started/previous-versions/
- torchao compatibility: https://github.com/pytorch/ao/issues/2919
- Unsloth docs: https://docs.unsloth.ai/

### Useful Commands
```bash
# Clean environment rebuild
conda env remove -n mtgenv -y
conda env create -f environment.yml
bash fix_environment.sh

# Quick version check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"

# Test full stack
python -c "from unsloth import FastLanguageModel; print('OK')"
```

---

## Files Created During Troubleshooting

1. **environment.yml** - Minimal conda environment (Python + pip only)
2. **setup_env.sh** - Initial setup script (had issues, replaced)
3. **fix_environment.sh** - ✅ Working installation script with proper version pinning
4. **reinstall_pytorch.sh** - Intermediate attempt (partial fix)

**Use:** `fix_environment.sh` is the final, working solution.

---

## Future Considerations

1. **Monitor Unsloth Updates**: Check if PyTorch 2.6+ support is added
2. **Pin All Versions**: Consider creating a complete `requirements.txt` with all versions pinned
3. **Test Before Upgrading**: Always test PyTorch upgrades in a separate environment first
4. **Model Selection**: Verify tokenizer compatibility before starting multi-day training runs
5. **Use uv Everywhere**: Significant speed improvement over pip/conda for package installation

---

## Time Saved by Learning These Lessons

- **Initial issue identification**: ~2 hours debugging version mismatches
- **torchao discovery**: ~30 minutes finding compatibility info
- **Environment rebuilds**: ~3-4 full rebuilds to get versions right
- **Total time**: ~4-5 hours of troubleshooting

**Next time:** Run `fix_environment.sh` and be running in ~10 minutes.

---

## Contact & References

- Unsloth GitHub: https://github.com/unslothai/unsloth
- Unsloth Docs: https://docs.unsloth.ai/
- PyTorch Download: https://download.pytorch.org/whl/
- torchao Releases: https://github.com/pytorch/ao/releases
