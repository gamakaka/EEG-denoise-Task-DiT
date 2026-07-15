from src.models.noise_predictors import TimeMixerNoisePredictor


def build_noise_predictor(config):
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    feats = model_cfg["feats"]

    if name not in {"timemixer", "time_mixer"}:
        raise ValueError(f"当前源码只保留 TimeMixer 噪声预测网络，不再支持: {model_cfg['name']}")

    return TimeMixerNoisePredictor(
        feats=feats,
        task_condition=model_cfg.get("task_condition", False),
        spectral_prototype_path=model_cfg.get("spectral_prototype_path"),
        prototype_cond_dim=model_cfg.get("prototype_cond_dim", feats),
        num_classes=model_cfg.get("num_classes", 2),
        condition_residual_scale=model_cfg.get("condition_residual_scale", 0.1),
        fusion_type=model_cfg.get("fusion_type", "fusion1"),
    )
