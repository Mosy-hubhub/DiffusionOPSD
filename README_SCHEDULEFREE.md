# Schedule-Free ReFL on Z-Image-Turbo

This branch reproduces the Z-Image-Turbo / HPSv3 ReFL experiments in *Large Diffusion Models Need More Than EMA*, Appendix A.5 (Table 5). HPSv3 is the only training reward. Schedule-Free support is added to the Z-Image ReFL trainer; the SD3.5-M trainer is unchanged.

| Paper run | Optimizer | Learning rate | Warmup updates | Evaluation weights |
|---|---|---:|---:|---|
| AdamW baseline | AdamW | `1e-5` | 0 | EMA LoRA, decay 0.9 |
| Schedule-Free 4x | SF-AdamW | `4e-5` | 5 | Averaged x LoRA |
| Schedule-Free 20x | SF-AdamW | `2e-4` | 5 | Averaged x LoRA |

All three use 100 updates, 48 prompt groups x 12 trajectories = 576 trajectories per update, six policy GPUs plus one reward GPU, micro-batch size 1 and gradient accumulation 96 per policy rank. Generation uses 1024 x 1024, nine steps, CFG 0 and bfloat16. The baseline is ReFL trained with AdamW and HPSv3; the two Schedule-Free runs use the same ReFL trainer and reward. The base model without RL is an additional evaluation reference.

## Run the paper configurations

Follow the installation instructions in the original [README](README.md) and its [HPSv3 reward-model setup](README.md#reward-model-setup), including the HPSv3 dependencies. The project installs the minimal local `schedulefree` package; do not substitute another implementation when reproducing the paper. Prepare the prompt split and set the shared topology:

```bash
python scripts/prepare_pickapic_prompts.py
export MODEL=Tongyi-MAI/Z-Image-Turbo
# Optional local/offline overrides; leave unset for upstream loading defaults.
# export MODEL=/path/to/Z-Image-Turbo
# export HPSV3_CHECKPOINT=/path/to/HPSv3.safetensors
# export HPSV3_CONFIG=/path/to/HPSv3_7B_local.yaml
export ZIMAGE_HEAVY_BRIDGE=0
export ZIMAGE_HEAVY_DIFF_BRIDGE=1
export PUBLIC_POLICY_WORLD_SIZE=6
export PUBLIC_LAUNCH_WORLD_SIZE=7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME WANDB_DISABLED

python scripts/check_reward_setup.py --backbone zimage --reward hpsv3
python scripts/smoke_reward_gradient.py --reward hpsv3
```

Model loading follows the upstream defaults: Z-Image-Turbo resolves from `Tongyi-MAI/Z-Image-Turbo`; HPSv3 resolves its pinned `MizzenAI/HPSv3` checkpoint and default Qwen2-VL configuration when `HPSV3_CHECKPOINT` and `HPSV3_CONFIG` are unset. Prompt paths and default output directories are relative to the repository. Local checkpoint paths and the Hugging Face cache remain optional environment/configuration overrides.

Log in with `wandb login` once, or set `WANDB_MODE=offline`. Run the following jobs sequentially on the same GPUs:

```bash
# AdamW baseline
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=7 \
  scripts/train_refl_zimage.py --config=config/refl.py:zimage_hpsv3 \
  --config.pretrained.model="$MODEL" \
  --config.refl.optimizer=adamw \
  --config.refl.learning_rate=1e-5 \
  --config.train.learning_rate=1e-5 \
  --config.refl.adamw_warmup_steps=0 \
  --config.run_name=refl_hpsv3_adamw \
  --config.save_dir=outputs/refl/refl_hpsv3_adamw

# Schedule-Free 4x
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=7 \
  scripts/train_refl_zimage.py --config=config/refl.py:zimage_hpsv3 \
  --config.pretrained.model="$MODEL" \
  --config.refl.optimizer=adamw_schedulefree \
  --config.refl.learning_rate=4e-5 \
  --config.train.learning_rate=4e-5 \
  --config.refl.schedulefree_warmup_steps=5 \
  --config.run_name=refl_hpsv3_sf4x \
  --config.save_dir=outputs/refl/refl_hpsv3_sf4x

# Schedule-Free 20x
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=7 \
  scripts/train_refl_zimage.py --config=config/refl.py:zimage_hpsv3 \
  --config.pretrained.model="$MODEL" \
  --config.refl.optimizer=adamw_schedulefree \
  --config.refl.learning_rate=2e-4 \
  --config.train.learning_rate=2e-4 \
  --config.refl.schedulefree_warmup_steps=5 \
  --config.run_name=refl_hpsv3_sf20x \
  --config.save_dir=outputs/refl/refl_hpsv3_sf20x
```

All runs load `config/refl.py:zimage_hpsv3` directly. The explicit optimizer, LR and warmup flags select the experiment; run names and output directories keep their artifacts separate.

## Added optimizer parameters

Override these with `--config.refl.<parameter>=<value>` when using `config/refl.py`.

| Parameter | Default in refl.py | Meaning |
|---|---|---|
| `optimizer` | `adamw` | Select `adamw` or `adamw_schedulefree`. |
| `adamw_warmup_steps` | `0` | Linear warmup length for plain AdamW, in optimizer updates. |
| `schedulefree_warmup_steps` | `0` | Internal SF warmup length; the two paper SF runs use 5. |
| `schedulefree_r` | `0.0` | Polynomial exponent in averaging weights `(k+1)^r * lr_max^weight_lr_power`. |
| `schedulefree_weight_lr_power` | `2.0` | LR exponent in those averaging weights. |
| `schedulefree_inner_momentum` | `0.0` | Optional inner first-moment momentum; disabled in the paper. |

`refl.learning_rate` controls the optimizer LR; keep `train.learning_rate` equal when overriding it. SF trains at interpolation y, updates internal z, and evaluates averaged x, without an additional EMA. Checkpoints contain LoRA and optimizer/trainer state, not the complete base-model weights. Final evaluation retains 1,000 PNGs for additional scoring.

**HPSv3 compatibility:** the paper experiments used the historical `_preprocess` image-processing route. This branch preserves that adapter; the newer upstream `preprocess_tensor` route is not assumed equivalent.

## Reproducibility and resume

The fixed settings include LoRA rank 32 / alpha 64, Adam betas (0.9, 0.999), epsilon `1e-8`, weight decay 0.01, gradient clipping 1.0, and loss `1e-3 * relu(2 - standardized_reward)`. Reward calibration uses 512 training prompts. Online evaluation uses 128 DrawBench prompts at update 0 and every 10 updates. Checkpoints are saved every 10 updates before online evaluation.

To resume, keep the whole run directory, including calibration, manifests, metrics, and checkpoint files. Keep the same optimizer settings and topology, keep 100 as the total update count, and start a new W&B run:

```bash
unset WANDB_RUN_ID WANDB_RESUME
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=7 \
  scripts/train_refl_zimage.py --config=config/refl.py:zimage_hpsv3 \
  --config.pretrained.model="$MODEL" \
  --config.refl.optimizer=adamw_schedulefree \
  --config.refl.learning_rate=4e-5 \
  --config.train.learning_rate=4e-5 \
  --config.refl.schedulefree_warmup_steps=5 \
  --config.save_dir=/path/to/existing/run \
  --config.resume_from=/path/to/existing/run/checkpoints/checkpoint-80 \
  --config.run_name=refl_hpsv3_sf4x_resume80
```

AdamW saves evaluation EMA LoRA in `lora/` and online LoRA in `resume_lora/`. SF saves averaged x in `lora/` and `ema.pt` (the historical filename, not an EMA); `optimizer.pt` contains z, second-moment and averaging/warmup state. Both restore per-policy-rank RNG state and trainer progress. Keep the same code/config when resuming; checkpoint provenance checks remain active.

## Evaluation protocol and paper results

All rows use the same 1,000 DrawBench prompts, 1024 x 1024 resolution, nine sampling steps, CFG 0, bfloat16, seed 42 plus the global generation-batch index, and batch size 8. Fine-tuned rows use checkpoint-100 `lora/`; the base reference uses no LoRA. Final training-evaluation PNGs are retained in `<save_dir>/final_eval/images/`.

The table reports scores from saved 8-bit PNGs. HPSv3 is the sole training reward; all other columns are evaluation only. PickScore uses the raw scale (training scale multiplied by 26), and `hpsv2` denotes HPSv2.1. HPSv3++ reports mu; UniPercept reports aesthetics (IAA), quality (IQA), and structure/texture richness (ISTA). The additional standalone scoring tools are not bundled; HPSv3++ and UniPercept require their external implementations and isolated environments.

| Run | HPSv3 | Pick | CLIP | HPSv2.1 | Aes | ImgR | HPSv3++ | Uni IAA | Uni IQA | Uni ISTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Z-Image base (no RL) | 6.2544 | 22.8690 | 0.2754 | 0.2962 | 5.4118 | 0.9947 | 12.0641 | 60.9511 | 75.0656 | 37.7925 |
| AdamW 1x, warmup 0 | 6.3500 | 22.8766 | 0.2754 | 0.2971 | 5.4177 | 0.9959 | 12.1271 | 60.7050 | 74.8078 | 38.3209 |
| SF-AdamW 4x, warmup 5 | 6.8626 | 22.8921 | 0.2748 | 0.2995 | 5.4456 | 1.0150 | 12.3485 | 60.6995 | 75.2258 | 38.3189 |
| SF-AdamW 20x, warmup 5 | 9.6134 | 22.8341 | 0.2807 | 0.3123 | 5.7066 | 1.1197 | 12.0347 | 51.5663 | 67.4061 | 41.7878 |

Reported values are from Appendix A.5, Table 5. The paper's ReFL qualitative comparisons are [Figure 10](assets/refl/figure_10.png), [Figure 11](assets/refl/figure_11.png), [Figure 12](assets/refl/figure_12.png), and [Figure 13](assets/refl/figure_13.png), with rows base / AdamW 1x / SF 4x / SF 20x.
