from pathlib import Path
import math

import torch


def experiment_variant(config):
    return "conditional" if config["model"].get("task_condition", False) else "unconditional"


def loss_variant(config):
    train_config = config["train"]
    content = float(train_config.get("lambda_content", 0.0))
    style = float(train_config.get("lambda_style", 0.0))
    if not math.isfinite(content) or not math.isfinite(style):
        raise ValueError("辅助损失权重必须为有限数值")
    if content < 0 or style < 0:
        raise ValueError("辅助损失权重不能小于 0")
    return f"content_{content:g}_style_{style:g}"


def checkpoint_paths(config):
    folder = (
        Path(config["output"]["checkpoint_root"])
        / experiment_variant(config)
        / loss_variant(config)
    )
    return {
        "best": folder / "best.pth",
        "final": folder / "final.pth",
        "epochs": folder / "epochs",
    }


def log_dir(config, stage):
    if stage not in {"train", "test"}:
        raise ValueError(f"不支持的日志阶段: {stage}")
    return (
        Path(config["output"]["log_root"])
        / stage
        / experiment_variant(config)
        / loss_variant(config)
    )


def evaluation_root(config):
    return (
        Path(config["output"]["result_root"])
        / experiment_variant(config)
        / loss_variant(config)
    )


def sampling_variant(config):
    eval_cfg = config["eval"]
    sampler = eval_cfg.get("sampler", "ddpm").lower()
    if sampler == "ddpm":
        sample_steps = config["diffusion"]["q_num_steps"]
    else:
        sample_steps = eval_cfg.get("ddim_num_steps", config["diffusion"]["q_num_steps"])
    eta = eval_cfg.get("ddim_eta", 0.0)
    return f"{sampler}_steps_{sample_steps}_eta_{eta:g}"


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    # Old unconditional/gated checkpoints contain this now-removed, unused table.
    checkpoint.pop("label_embed.weight", None)
    model.model.load_state_dict(checkpoint)
    return model
