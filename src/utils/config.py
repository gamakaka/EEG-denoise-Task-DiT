import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config, args):
    if args.device is not None:
        config["project"]["device"] = args.device
    if hasattr(args, "conditional") and args.conditional is not None:
        config["model"]["task_condition"] = args.conditional
    if hasattr(args, "fusion_type") and args.fusion_type is not None:
        config["model"]["fusion_type"] = args.fusion_type
    if hasattr(args, "lambda_content") and args.lambda_content is not None:
        config["train"]["lambda_content"] = args.lambda_content
    if hasattr(args, "lambda_style") and args.lambda_style is not None:
        config["train"]["lambda_style"] = args.lambda_style
    if (
        hasattr(args, "encoder_ckpt_path")
        and args.encoder_ckpt_path is not None
    ):
        config["train"]["encoder_ckpt_path"] = args.encoder_ckpt_path
    if hasattr(args, "sampler") and args.sampler is not None:
        config["eval"]["sampler"] = args.sampler
    if hasattr(args, "ddim_num_steps") and args.ddim_num_steps is not None:
        config["eval"]["ddim_num_steps"] = args.ddim_num_steps
    if hasattr(args, "ddim_eta") and args.ddim_eta is not None:
        config["eval"]["ddim_eta"] = args.ddim_eta
    return config
