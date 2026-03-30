# LIBERO Fine-Tuning Setup Steps

## Environment
- GPU: NVIDIA L4 (23GB VRAM)
- RAM: 16GB (swap required — see step 5b)
- Config: `pi0_libero_low_mem_finetune` (LoRA, fits in <24GB)
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

### 3. Fetch LIBERO dataset

Dataset: `physical-intelligence/libero` (1693 episodes, ~273k frames, ~35GB)

**Plan A: Fetch from S3 (faster, no HF token needed)**
```bash
aws s3 sync s3://chris-purina-playground/openpi/cache/huggingface/lerobot/physical-intelligence/libero/ \
  ~/.cache/huggingface/lerobot/physical-intelligence/libero/
```

**Plan B: Download from HuggingFace (fallback if S3 copy is missing/stale)**

This happens automatically when running `compute_norm_stats.py`, but requires a HF token with access to the gated repo:
```bash
huggingface-cli login --token <HF_TOKEN>
```

### 4. Compute normalization statistics
```bash
uv run python scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune
```
Completed 2026-03-27. 8545 batches, ~63 min on 2 CPU workers. Output: `assets/pi0_libero_low_mem_finetune/physical-intelligence/libero/norm_stats.json`

**Norm stats also backed up to S3:**
```bash
# Upload
aws s3 cp assets/pi0_libero_low_mem_finetune/physical-intelligence/libero/norm_stats.json \
  s3://chris-purina-playground/openpi/assets/pi0_libero_low_mem_finetune/physical-intelligence/libero/norm_stats.json

# Restore (skip recompute)
aws s3 cp s3://chris-purina-playground/openpi/assets/pi0_libero_low_mem_finetune/physical-intelligence/libero/norm_stats.json \
  assets/pi0_libero_low_mem_finetune/physical-intelligence/libero/norm_stats.json
```

### 5. Install system dependencies

#### 5a. Build tools and libraries
```bash
sudo apt-get install -y gcc build-essential libgl1 libglib2.0-0
```
Required for `evdev` (C compiler) and `opencv` (`libGL`).

#### 5b. NVIDIA driver (if not already loaded)
The L4 GPU needs a driver installed. Without it, JAX falls back to CPU and training OOMs on 16GB RAM.
```bash
sudo apt-get install -y nvidia-driver-550-server
sudo modprobe nvidia
nvidia-smi  # verify: should show L4, CUDA 12.8
```
Installed 2026-03-30. Driver 570.211.01 was pulled in (supersedes 550). No reboot needed — `modprobe` sufficed.

#### 5c. Add swap (prevents OOM kills)
16GB RAM is not enough for dataset loading. Without swap, the OOM killer terminates the training process (and the shell session).
```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```
Note: swap does not persist across reboots. Re-run `swapon` after restart, or add to `/etc/fstab`.

### 6. Training command (run manually)
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/train.py pi0_libero_low_mem_finetune --exp-name=libero_lora
```

Optional: add `--overwrite` to overwrite existing experiment, or use wandb with `WANDB_API_KEY=<key>`.

## Notes
- LoRA fine-tuning config freezes most parameters, trains only LoRA adapters
- EMA is disabled for LoRA fine-tuning
- 30,000 training steps by default (override with `--num-train-steps=N`)
- Base checkpoint auto-downloaded from `gs://openpi-assets/checkpoints/pi0_base/params`
- Norm stats saved to project assets directory after computation
- 2026-03-30: First training attempts OOM-killed the shell (no GPU driver → JAX used CPU → 16GB RAM exhausted). Fixed by installing NVIDIA driver and adding 16GB swap.
