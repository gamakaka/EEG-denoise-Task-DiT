from pathlib import Path

import torch
import torch.nn as nn

from comparisons.rescnn1d.model import OneDResCNN


class FrozenEEGEncoder(nn.Module):
    """Frozen 1D-ResCNN encoder used only for perceptual content loss."""

    def __init__(self, checkpoint_path):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"EEG Encoder checkpoint 不存在: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "encoder_state_dict" not in checkpoint:
            raise ValueError(
                "EEG Encoder checkpoint 必须包含 encoder_state_dict: "
                f"{checkpoint_path}"
            )
        required_metadata = {"signal_length", "feature_dim", "feature_length", "feature_shape"}
        missing_metadata = sorted(required_metadata - checkpoint.keys())
        if missing_metadata:
            raise ValueError(
                "EEG Encoder checkpoint 是不再支持的旧版本，缺少新 Encoder "
                f"元数据 {missing_metadata}: {checkpoint_path}"
            )

        model_config = checkpoint.get("config", {}).get("model", {})
        channels = tuple(model_config.get("channels", (32, 64, 128, 64, 32)))
        input_channels = int(model_config.get("input_channels", 1))
        output_channels = int(model_config.get("output_channels", 1))
        kernel_size = int(model_config.get("kernel_size", 3))
        self.signal_length = int(checkpoint["signal_length"])
        self.feature_dim = int(checkpoint["feature_dim"])
        self.feature_length = int(checkpoint["feature_length"])
        feature_shape = tuple(int(value) for value in checkpoint["feature_shape"])
        expected_shape = (self.feature_dim, self.feature_length)
        if feature_shape != expected_shape:
            raise ValueError(
                f"checkpoint feature_shape={feature_shape} 与元数据 "
                f"{expected_shape} 不一致"
            )
        if self.feature_dim != channels[-1]:
            raise ValueError(
                f"checkpoint feature_dim={self.feature_dim} 与 Encoder 最后一层 "
                f"通道数 {channels[-1]} 不一致"
            )
        if self.feature_length >= self.signal_length:
            raise ValueError(
                "新 EEG Encoder 必须输出压缩后的时间维度，"
                f"实际 signal_length={self.signal_length}, "
                f"feature_length={self.feature_length}"
            )
        encoder_model = OneDResCNN(
            input_channels=input_channels,
            output_channels=output_channels,
            channels=channels,
            kernel_size=kernel_size,
            signal_length=self.signal_length,
            feature_length=self.feature_length,
            use_label=False,
        )
        self.encoder = encoder_model.encoder
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # The parent DDPM enters train mode each epoch; this encoder must not.
        super().train(False)
        return self

    def forward(self, eeg):
        if eeg.ndim != 3 or eeg.shape[1] != 1:
            raise ValueError(
                f"EEG Encoder 输入必须为 [B, 1, T]，实际为 {tuple(eeg.shape)}"
            )
        if eeg.shape[-1] != self.signal_length:
            raise ValueError(
                f"EEG Encoder 要求信号长度 {self.signal_length}，"
                f"实际为 {eeg.shape[-1]}"
            )
        features = self.encoder(eeg)
        expected = (self.feature_dim, self.feature_length)
        if tuple(features.shape[1:]) != expected:
            raise RuntimeError(
                f"EEG Encoder 输出应为 [B, {self.feature_dim}, "
                f"{self.feature_length}]，实际为 {tuple(features.shape)}"
            )
        return features
