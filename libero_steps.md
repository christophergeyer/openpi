# LIBERO Fine-Tuning Setup Steps

## Environment
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- Config: `pi0_libero_low_mem_finetune` (LoRA, fits in <32GB)
- Framework: JAX (openpi native)

## Commands Run

### 1. Install dependencies
```bash
cd /root/cloud_setup/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 2. Verify config loads
```bash
uv run python -c "from openpi.training.config import get_config; c = get_config('pi0_libero_low_mem_finetune'); print(c.name)"
```

### 3. Compute normalization statistics (also downloads LIBERO dataset)
Dataset: `physical-intelligence/libero` (1693 episodes, ~273k frames)
Downloaded to: `/workspace/.cache/huggingface/lerobot/physical-intelligence/libero/`
```bash
uv run python scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune
```

### 4. Training command (run manually)
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/train.py pi0_libero_low_mem_finetune --exp-name=libero_lora
```

Optional: add `--overwrite` to overwrite existing experiment, or use wandb with `WANDB_API_KEY=<key>`.

## Notes
- LoRA fine-tuning config freezes most parameters, trains only LoRA adapters
- EMA is disabled for LoRA fine-tuning
- 30,000 training steps by default
- Base checkpoint auto-downloaded from `gs://openpi-assets/checkpoints/pi0_base/params`
- Norm stats saved to project assets directory after computation
