"""The opt-in DiffusionOPSD v2 recipe, shared with the public release."""

from ml_collections import ConfigDict


TRAJECTORIES_PER_UPDATE = {"sd3": 1152, "zimage": 576}


def apply_opsd_v2(config: ConfigDict, *, backbone: str, policy_world_size: int) -> ConfigDict:
    """Resolve the single v2 switch before constructing loaders or logging config.

    The world size counts policy workers only, excluding reward-server ranks.
    Preserve the configured microbatch and adjust accumulation to the recipe's
    trajectory budget. Calling this again produces the same configuration.
    """
    if not bool(getattr(getattr(config, "opsd", None), "v2", False)):
        return config
    if backbone not in TRAJECTORIES_PER_UPDATE:
        raise ValueError(f"Unsupported OPSD v2 backbone: {backbone}")
    if not bool(config.opsd.opa):
        raise ValueError("OPSD v2 requires an OPSD/OPA configuration.")
    microbatch = int(config.sample.train_batch_size)
    world_size = int(policy_world_size)
    total = TRAJECTORIES_PER_UPDATE[backbone]
    if world_size <= 0 or microbatch <= 0 or total % (world_size * microbatch):
        raise ValueError(f"OPSD v2 budget {total} must divide policy world size × microbatch.")
    batches = total // (world_size * microbatch)
    config.sample.num_image_per_prompt = 1
    config.sample.num_batches_per_epoch = batches
    config.train.batch_size = microbatch
    config.train.gradient_accumulation_steps = batches
    config.train.num_inner_epochs = 1
    config.train.adv_mode = "neutral"
    config.per_prompt_stat_tracking = False
    config.opsd.opa_n_ascent = 1
    config.decay_type = 1
    config.train.ema = True
    config.efficiency = ConfigDict({
        "query_prefix": True, "cache_old_query": False, "ddp_no_sync": False,
        "cache_sd3_text": False, "cache_clip_text": False,
    })
    validate_query_prefix(config, backbone=backbone)
    return config


def validate_query_prefix(config: ConfigDict, *, backbone: str) -> bool:
    """Reject recipes that still require the endpoint discarded by prefix rollout."""
    if not bool(getattr(getattr(config, "efficiency", None), "query_prefix", False)):
        return False
    checks = {
        "OPA": bool(config.opsd.opa),
        "neutral K=1": (config.sample.num_image_per_prompt == 1
                        and config.train.adv_mode == "neutral"),
        "rollout query": (config.opsd.train_state == "rollout"
                          and config.opsd.opa_state_mode == "rollout"),
        "replacement targets": config.opsd.opa_target_mode == "replace",
        "one reward-gradient step": (config.opsd.opa_n_ascent == 1
                                     and config.opsd.opa_dir_mode == "grad"),
        "no endpoint certification or refinement": (not config.opsd.opa_cert
                                                    and not config.opsd.refine),
    }
    if backbone == "sd3":
        checks["deterministic DPM2"] = config.sample.solver == "dpm2" and config.sample.deterministic
    elif backbone == "zimage":
        checks["native Euler without CFG"] = config.sample.guidance_scale == 0.0
    else:
        raise ValueError(f"Unsupported query-prefix backbone: {backbone}")
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError("Query-prefix rollout requires: " + ", ".join(failed))
    return True
