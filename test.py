import argparse

from torch.utils.data import DataLoader

from src.data import EEGDataset
from src.evaluate import evaluate_ddpm
from src.models import DDPM, build_noise_predictor
from src.utils import (
    apply_overrides,
    build_logger,
    checkpoint_paths,
    experiment_variant,
    load_checkpoint,
    load_config,
    log_dir,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--conditional",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用或关闭任务频谱原型条件",
    )
    parser.add_argument("--lambda-content", type=float, default=None)
    parser.add_argument("--lambda-style", type=float, default=None)
    parser.add_argument(
        "--fusion-type",
        type=str,
        default=None,
        choices=["fusion1", "fusion2", "fusion3", "fusion4"],
    )
    parser.add_argument("--sampler", type=str, default=None, choices=["ddpm", "ddim"])
    parser.add_argument("--ddim-num-steps", dest="ddim_num_steps", type=int, default=None)
    parser.add_argument("--ddim-eta", dest="ddim_eta", type=float, default=None)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    set_seed(config["project"]["seed"])

    variant = experiment_variant(config)
    logger = build_logger(f"test.{variant}", log_dir(config, "test"), "test")
    logger.info(f"模型: {config['model']['name']}")
    logger.info(f"融合模块: {config['model'].get('fusion_type', 'fusion1')}")
    logger.info(f"实验模式: {variant}")
    logger.info(
        f"训练辅助损失权重: content={config['train']['lambda_content']}, "
        f"style={config['train']['lambda_style']}"
    )

    device = config["project"]["device"]
    base_model = build_noise_predictor(config).to(device)
    model = DDPM(
        base_model,
        config,
        device,
        eps_scaler=config["model"]["eps_scaler"],
    )

    paths = checkpoint_paths(config)
    load_checkpoint(model, paths["best"])
    logger.info(f"加载模型: {paths['best']}")

    test_sets = {
        "rest": EEGDataset(config["data"]["test_rest_path"]),
        "motion": EEGDataset(config["data"]["test_motion_path"]),
    }
    for eeg_type, dataset in test_sets.items():
        test_loader = DataLoader(
            dataset,
            batch_size=config["eval"]["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=config["data"]["num_workers"],
        )
        logger.info("=" * 30 + f" Test {eeg_type} " + "=" * 30)
        evaluate_ddpm(model, test_loader, config, logger, eeg_type=eeg_type)


if __name__ == "__main__":
    main()
