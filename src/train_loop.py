from pathlib import Path

import torch
from torch.optim import Adam


def train_ddpm(model, config, train_loader, val_loader, checkpoint_paths, logger):
    train_cfg = config["train"]
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg["scheduler_step_size"],
        gamma=train_cfg["scheduler_gamma"],
    )

    best_valid_loss = float("inf")
    patience = 0
    best_path = checkpoint_paths["best"]
    final_path = checkpoint_paths["final"]
    epoch_dir = checkpoint_paths["epochs"]
    Path(best_path).parent.mkdir(parents=True, exist_ok=True)
    if train_cfg.get("save_epoch_checkpoints", False):
        Path(epoch_dir).mkdir(parents=True, exist_ok=True)

    for epoch_no in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        train_components = {"ddpm": 0.0, "content": 0.0, "style": 0.0}
        batch_count = 0
        for batch_no, batch in enumerate(train_loader, start=1):
            batch_count = batch_no
            clean_batch = batch["clean_eeg"].to(model.device)
            noisy_batch = batch["noisy_eeg"].to(model.device)
            label = batch["label"].to(model.device)

            optimizer.zero_grad()
            loss = model(clean_batch, noisy_batch, label)
            loss.backward()
            if train_cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            train_loss += loss.item()
            for name in train_components:
                train_components[name] += model.loss_components[name].item()

        if batch_count == 0:
            raise ValueError("训练 DataLoader 为空")
        train_loss /= batch_count
        for name in train_components:
            train_components[name] /= batch_count
        logger.info(
            f"epoch={epoch_no} train "
            f"total={train_loss:.6e} "
            f"ddpm={train_components['ddpm']:.6e} "
            f"content={train_components['content']:.6e} "
            f"style={train_components['style']:.6e}"
        )
        scheduler.step()

        val_loss, val_components = validate_ddpm(model, val_loader)
        logger.info(
            f"epoch={epoch_no} val "
            f"total={val_loss:.6e} "
            f"ddpm={val_components['ddpm']:.6e} "
            f"content={val_components['content']:.6e} "
            f"style={val_components['style']:.6e}"
        )
        if train_cfg.get("save_epoch_checkpoints", False):
            epoch_path = epoch_dir / f"epoch_{epoch_no + 1:04d}.pth"
            torch.save(model.model.state_dict(), epoch_path)
            logger.info(f"保存 epoch 模型: {epoch_path}")
        if best_valid_loss - val_loss > train_cfg["min_delta"]:
            best_valid_loss = val_loss
            patience = 0
            torch.save(model.model.state_dict(), best_path)
            logger.info(f"保存最佳模型: {best_path}")
        else:
            patience += 1

        if patience > train_cfg["early_stop_patience"]:
            logger.info("触发早停")
            break

    torch.save(model.model.state_dict(), final_path)
    logger.info(f"保存最终模型: {final_path}")


@torch.no_grad()
def validate_ddpm(model, val_loader):
    model.eval()
    total_loss = 0.0
    components = {"ddpm": 0.0, "content": 0.0, "style": 0.0}
    batch_count = 0
    for batch_no, batch in enumerate(val_loader, start=1):
        batch_count = batch_no
        clean_batch = batch["clean_eeg"].to(model.device)
        noisy_batch = batch["noisy_eeg"].to(model.device)
        label = batch["label"].to(model.device)
        total_loss += model(clean_batch, noisy_batch, label).item()
        for name in components:
            components[name] += model.loss_components[name].item()
    if batch_count == 0:
        raise ValueError("验证 DataLoader 为空")
    for name in components:
        components[name] /= batch_count
    return total_loss / batch_count, components
