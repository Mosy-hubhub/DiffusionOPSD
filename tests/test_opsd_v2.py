"""CPU contracts for the default v2 recipe and native sampler prefixes."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from ml_collections import ConfigDict

from diffusionopsd.diffusers_patch.solver import run_sampling
from diffusionopsd.diffusers_patch.zimage_pipeline_with_rollout import (
    zimage_query_sigma_index,
    zimage_rollout,
)
from diffusionopsd.experiment_io import write_raw_reward_jsonl
from diffusionopsd.opsd_v2 import apply_opsd_v2, validate_query_prefix


ROOT = Path(__file__).resolve().parents[1]
REWARDS = ("hpsv2", "clipscore", "pickscore", "aesthetic", "imagereward", "hpsv3", "deqa")


def load_config(name):
    spec = importlib.util.spec_from_file_location("public_config", ROOT / "config/public.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config(name)


def test_configs_without_opsd_remain_unchanged():
    config = ConfigDict({"seed": 42})
    apply_opsd_v2(config, backbone="sd3", policy_world_size=8)
    assert config.to_dict() == {"seed": 42}


@pytest.mark.parametrize("backbone,reward", [
    *(('sd35', r) for r in (*REWARDS, 'open3')),
    *(('zimage', r) for r in REWARDS),
])
def test_recipe_preserves_legacy_and_resolves_policy_budget(monkeypatch, backbone, reward):
    heavy = backbone == "zimage" and reward in {"hpsv3", "deqa"}
    world = 6 if heavy else 8
    monkeypatch.setenv("PUBLIC_POLICY_WORLD_SIZE", str(world))
    monkeypatch.setenv("PUBLIC_LAUNCH_WORLD_SIZE", str(world + int(heavy)))
    monkeypatch.setenv("ZIMAGE_HEAVY_DIFF_BRIDGE", str(int(heavy)))
    config = load_config(f"{backbone}_{reward}")
    assert config.opsd.v2 is True
    kind = "sd3" if backbone == "sd35" else "zimage"
    legacy = ConfigDict(config.to_dict())
    legacy.opsd.v2 = False
    legacy_before = legacy.to_dict()
    apply_opsd_v2(legacy, backbone=kind, policy_world_size=world)
    assert legacy.to_dict() == legacy_before
    assert legacy.opsd.opa_n_ascent == 2
    assert not validate_query_prefix(legacy, backbone=kind)
    before = config.to_dict()
    config.lock()  # ml_collections locks configs when applying CLI overrides.
    apply_opsd_v2(config, backbone=kind, policy_world_size=world)
    budget = 1152 if kind == "sd3" else 576
    assert world * config.sample.train_batch_size * config.sample.num_batches_per_epoch == budget
    assert world * config.train.batch_size * config.train.gradient_accumulation_steps == budget
    assert config.sample.num_image_per_prompt == 1
    assert config.opsd.opa_n_ascent == 1
    assert config.train.adv_mode == "neutral"
    assert not config.per_prompt_stat_tracking
    assert config.train.num_inner_epochs == 1
    assert config.train.ema and config.decay_type == 1
    for key in ("reward_fn", "resolution", "num_epochs", "seed", "pretrained"):
        assert config.to_dict()[key] == before[key]
    resolved = config.to_dict()
    apply_opsd_v2(config, backbone=kind, policy_world_size=world)
    assert config.to_dict() == resolved


@pytest.mark.parametrize("field,value", [
    ("opa_cert", 1), ("opa_state_mode", "forward"), ("opa_target_mode", "aux"),
    ("refine", 1), ("opa_dir_mode", "residual"),
])
def test_v2_rejects_endpoint_dependent_recipes(monkeypatch, field, value):
    monkeypatch.setenv("PUBLIC_POLICY_WORLD_SIZE", "8")
    config = load_config("sd35_hpsv2")
    config.opsd.v2 = True
    config.opsd[field] = value
    with pytest.raises(ValueError, match="Query-prefix"):
        apply_opsd_v2(config, backbone="sd3", policy_world_size=8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("stop", [0, 1, 8])
def test_dpm2_prefix_matches_query_and_rng_across_batches(dtype, stop):
    schedule = torch.linspace(1, 0, 11)
    paired = []
    for limit in (None, stop):
        torch.manual_seed(42)
        batches = []
        for _ in range(3):
            initial = torch.randn(2, 2, 4, 4, dtype=dtype)
            _, states, _ = run_sampling(lambda z, sigma: 0.3 * z + 0.2 * sigma,
                initial, schedule, solver="dpm2", determistic=True, max_steps=limit)
            batches.append((initial.clone(), states[stop].clone(), torch.get_rng_state()))
        paired.append(batches)
    for full, prefix in zip(*paired, strict=True):
        for a, b in zip(full, prefix, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_zimage_prefix_uses_native_scheduler_and_skips_endpoint():
    pipe = SimpleNamespace(transformer=object(),
                           scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0))
    initial = torch.linspace(-1, 1, 64).reshape(1, 1, 8, 8)
    embeds = [torch.ones(2, 3)]
    path = "diffusionopsd.diffusers_patch.zimage_pipeline_with_rollout._transformer_v_raw"
    with patch(path, side_effect=lambda model, z, t, prompts: 0.3 * z + t[:, None, None, None]) as velocity:
        full = zimage_rollout(pipe, embeds, 9, 64, 64, "cpu", decode=False, latents=initial)
        full_calls = velocity.call_count
        rng = torch.get_rng_state()
        velocity.reset_mock()
        prefix = zimage_rollout(pipe, embeds, 9, 64, 64, "cpu", decode=False,
                               latents=initial, stop_before_sigma=0.273)
        query = zimage_query_sigma_index(full["sigmas"], 0.273)
        assert velocity.call_count == query < full_calls
        assert prefix["completed_prefix_steps"] == query
        assert "x0" not in prefix and "images" not in prefix
        for key in ("sigmas", "timesteps"):
            torch.testing.assert_close(full[key], prefix[key], atol=0, rtol=0)
        torch.testing.assert_close(full["latents"][query], prefix["latents"][-1], atol=0, rtol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        with pytest.raises(ValueError, match="either"):
            zimage_rollout(pipe, embeds, 9, 64, 64, "cpu", decode=False, latents=initial,
                           stop_before_sigma=0.273, stop_before_index=query)


def test_anchor_reward_records_keep_scope_and_values(tmp_path):
    name = write_raw_reward_jsonl(str(tmp_path), epoch=0, global_step=0,
        prompts=["one", "two"], rewards={"hpsv3": np.array([15.5, 16.5])},
        reward_scope="query_anchor")
    records = [json.loads(line) for line in Path(name).read_text().splitlines()]
    assert [r["rewards_raw"]["hpsv3"] for r in records] == [15.5, 16.5]
    assert all(r["reward_scope"] == "query_anchor" for r in records)


@pytest.mark.parametrize("script,preset,kind", [
    ("train_opsd_ri_sd3.py", "sd35_hpsv2", "sd3"),
    ("train_opsd_zimage.py", "zimage_hpsv2", "zimage"),
])
def test_real_trainer_module_and_default_cli(script, preset, kind):
    # Import the complete trainer, then parse its native flags without invoking GPU main().
    code = '''
import runpy, sys
script, preset, kind = sys.argv[1:]
sys.argv = [script, "--config=config/public.py:" + preset]
namespace = runpy.run_path(script, run_name="opsd_import_contract")
flags = namespace["FLAGS"]
flags(sys.argv)
config = flags.config
namespace["apply_opsd_v2"](config, backbone=kind, policy_world_size=8)
assert config.opsd.v2 and config.opsd.opa_n_ascent == 1
assert config.sample.num_image_per_prompt == 1
assert namespace["validate_query_prefix"](config, backbone=kind)

# The actual target helper must reuse one gradient for both directions.
import torch
from unittest.mock import patch
target_step = namespace["_opa_tr_step"]
anchor = torch.linspace(0.1, 1, 32).reshape(2, 1, 4, 4).requires_grad_(True)
reward = anchor.square().flatten(1).mean(1)
gradient, = torch.autograd.grad(reward.sum(), anchor)
with patch.dict(target_step.__globals__, {"_reward_of_latents_grad": lambda *a, **k: (_ for _ in ()).throw(AssertionError("extra reward forward"))}):
    targets = [target_step(None, None, "hpsv2", anchor, ["a", "b"], 0.1,
               config.opsd.opa_n_ascent, 1.0, direction, first_grad=gradient)
               for direction in (1.0, -1.0)]
assert not targets[0].requires_grad and not targets[1].requires_grad
torch.testing.assert_close(targets[0] - anchor, anchor - targets[1], atol=1e-7, rtol=1e-6)
assert torch.all((targets[0] - anchor).flatten(1).norm(dim=1) > 0)
print("native trainer import and CLI passed")
'''
    env = dict(os.environ, PUBLIC_POLICY_WORLD_SIZE="8", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts" / script), preset, kind],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr[-6000:]
