from copy import deepcopy
from math import log

import torch
import torch.nn as nn

from timemixer import config as timemixer_config
from timemixer.models import TimeMixer


class Fusion1(nn.Module):
    """Original local-global feature fusion used by the TimeMixer predictor."""

    def __init__(self, dim, r=2):
        super(Fusion1, self).__init__()
        self.local_att = nn.Sequential(
            nn.Conv1d(dim, dim // r, kernel_size=1),
            nn.GroupNorm(1, dim // r, eps=1e-08),
            nn.ReLU(inplace=True),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // r, kernel_size=1),
            nn.GroupNorm(1, dim // r, eps=1e-08),
            nn.ReLU(inplace=True),
        )

        self.p1 = nn.Sequential(
            nn.Conv1d(dim // r, dim, kernel_size=1),
            nn.GroupNorm(1, dim, eps=1e-08),
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(dim // r, dim, kernel_size=1),
            nn.GroupNorm(1, dim, eps=1e-08),
        )

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        return x * self.p1(xlg) + residual * self.p2(xlg)


def _valid_num_heads(dim, preferred=4):
    for heads in range(min(preferred, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class Fusion2(nn.Module):
    """Dual-branch self-attention fusion without gates."""

    def __init__(self, dim, r=2, num_heads=1):
        super().__init__()
        hidden = max(1, dim // r)
        heads = _valid_num_heads(hidden, num_heads)
        self.x_pre = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden, eps=1e-08),
            nn.SiLU(),
        )
        self.residual_pre = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden, eps=1e-08),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            batch_first=True,
        )
        self.local_refine = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GroupNorm(1, hidden, eps=1e-08),
            nn.SiLU(),
        )
        self.out = nn.Sequential(
            nn.Conv1d(hidden, dim, kernel_size=1),
            nn.GroupNorm(1, dim, eps=1e-08),
        )

    def forward(self, x, residual):
        mixed = self.x_pre(x) + self.residual_pre(residual)
        tokens = self.norm(mixed.transpose(1, 2))
        attn, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        attn = attn.transpose(1, 2)
        context = self.local_refine(mixed + attn)
        return 0.5 * (x + residual) + self.out(context)



class Fusion3(nn.Module):
    """Local-global self-attention fusion over the mixed branch.

    This follows the spirit of Fusion1: first form a joint branch from x_t and
    noisy EEG, then use lightweight temporal self-attention to produce two
    branch weights for x and residual.
    """

    def __init__(self, dim, r=2, num_heads=1):
        super().__init__()
        hidden = max(1, dim // r)
        heads = _valid_num_heads(hidden, num_heads)
        self.pre = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden, eps=1e-08),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            batch_first=True,
        )
        self.local_att = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GroupNorm(1, hidden, eps=1e-08),
            nn.SiLU(),
        )
        self.p1 = nn.Sequential(
            nn.Conv1d(hidden, dim, kernel_size=1),
            nn.GroupNorm(1, dim, eps=1e-08),
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(hidden, dim, kernel_size=1),
            nn.GroupNorm(1, dim, eps=1e-08),
        )

    def forward(self, x, residual):
        mixed = self.pre(x + residual)
        tokens = self.norm(mixed.transpose(1, 2))
        attn, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        attn = attn.transpose(1, 2)
        context = self.local_att(mixed + attn)
        return x * self.p1(context) + residual * self.p2(context)


class Fusion4(nn.Module):
    """Sum-difference self-attention fusion without gates."""

    def __init__(self, dim, r=2, num_heads=1):
        super().__init__()
        hidden = max(1, dim // r)
        heads = _valid_num_heads(hidden, num_heads)
        self.sum_pre = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden, eps=1e-08),
            nn.SiLU(),
        )
        self.diff_pre = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden, eps=1e-08),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            batch_first=True,
        )
        self.local_att = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GroupNorm(1, hidden, eps=1e-08),
            nn.SiLU(),
        )
        self.x_out = nn.Sequential(
            nn.Conv1d(hidden, dim, kernel_size=1),
            nn.GroupNorm(1, dim, eps=1e-08),
        )
        self.residual_out = nn.Sequential(
            nn.Conv1d(hidden, dim, kernel_size=1),
            nn.GroupNorm(1, dim, eps=1e-08),
        )

    def forward(self, x, residual):
        mixed = self.sum_pre(x + residual) + self.diff_pre(x - residual)
        tokens = self.norm(mixed.transpose(1, 2))
        attn, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        attn = attn.transpose(1, 2)
        context = self.local_att(mixed + attn)
        branch_update = self.x_out(context) + self.residual_out(context)
        return 0.5 * (x + residual) + 0.5 * branch_update


FUSION_REGISTRY = {
    "fusion1": Fusion1,
    "fusion2": Fusion2,
    "fusion3": Fusion3,
    "fusion4": Fusion4,
}


def build_fusion(name, dim):
    key = str(name or "fusion1").lower()
    if key not in FUSION_REGISTRY:
        available = ", ".join(sorted(FUSION_REGISTRY))
        raise ValueError(f"未知 fusion_type={name}，可选: {available}")
    return FUSION_REGISTRY[key](dim=dim)


class TimestepEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t_scale):
        t_scale = t_scale.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=t_scale.dtype, device=t_scale.device) / count
        encoding = t_scale.unsqueeze(1) * torch.exp(-log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding.unsqueeze(-1)


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class TimeMixerNoisePredictor(nn.Module):
    """基于 TimeMixer 的 DDPM 高斯噪声预测网络。"""

    def __init__(
        self,
        feats=64,
        model_cfg=None,
        task_condition=False,
        spectral_prototype_path=None,
        prototype_cond_dim=None,
        num_classes=2,
        condition_residual_scale=0.1,
        fusion_type="fusion1",
    ):
        super().__init__()
        self.task_condition = task_condition
        self.num_classes = num_classes
        self.condition_residual_scale = condition_residual_scale
        self.fusion_type = fusion_type
        if condition_residual_scale < 0:
            raise ValueError("condition_residual_scale 不能小于 0")
        prototype_cond_dim = prototype_cond_dim or feats
        if prototype_cond_dim != feats:
            raise ValueError(
                f"prototype_cond_dim ({prototype_cond_dim}) 必须与条件维度 feats "
                f"({feats}) 一致"
            )

        cfg = deepcopy(model_cfg or timemixer_config.model_cfg)
        cfg.enc_in = feats
        cfg.dec_in = feats
        cfg.c_out = feats

        self.gamma = nn.Linear(feats, feats)
        self.beta = nn.Linear(feats, feats)
        self.time_embed = TimestepEncoding(feats)
        self.cond_fuse = nn.Sequential(nn.Linear(feats * 2, feats), nn.SiLU())
        if task_condition:
            self.task_norm = nn.LayerNorm(feats)
            self.condition_gamma = nn.Linear(feats, feats)
            self.condition_beta = nn.Linear(feats, feats)
            nn.init.zeros_(self.condition_gamma.weight)
            nn.init.zeros_(self.condition_gamma.bias)
            nn.init.zeros_(self.condition_beta.weight)
            nn.init.zeros_(self.condition_beta.bias)
            prototypes = self._load_spectral_prototypes(
                spectral_prototype_path,
                expected_num_classes=num_classes,
            )
            self.register_buffer("spectral_prototypes", prototypes)
            freq_dim = prototypes.shape[1]
            self.prototype_mlp = nn.Sequential(
                nn.Linear(freq_dim, prototype_cond_dim),
                nn.SiLU(),
                nn.Linear(prototype_cond_dim, feats),
            )

        self.x_encode = nn.Sequential(
            Conv1d(1, feats, 3, padding=1),
            Conv1d(feats, feats, 3, padding=1),
        )
        self.noisy_eeg_encode = nn.Sequential(
            Conv1d(1, feats, 3, padding=1),
            Conv1d(feats, feats, 3, padding=1),
        )

        self.fusion = build_fusion(fusion_type, dim=feats)
        self.timemixer = TimeMixer.Model(cfg)
        self.norm = nn.LayerNorm(feats)
        self.proj = nn.Sequential(
            nn.Linear(feats, feats * 2),
            nn.ReLU(),
            nn.Linear(feats * 2, 1),
        )

    def forward(self, x, noisy_eeg, label, t_scale):
        time_emb = self.time_embed(t_scale).squeeze(2)
        if self.task_condition:
            label = self._validate_labels(label)
            task_emb = self.prototype_mlp(self.spectral_prototypes[label])

        gamma = self.gamma(time_emb)
        beta = self.beta(time_emb)
        if self.task_condition:
            task_emb = self.task_norm(task_emb)
            task_context = self.cond_fuse(torch.cat([time_emb, task_emb], dim=-1))
            gamma = gamma + self.condition_residual_scale * torch.tanh(
                self.condition_gamma(task_context)
            )
            beta = beta + self.condition_residual_scale * torch.tanh(
                self.condition_beta(task_context)
            )

        x = self.x_encode(x)
        noisy_eeg = self.noisy_eeg_encode(noisy_eeg)
        x = self.fusion(x, noisy_eeg).permute(0, 2, 1)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = self.timemixer(x, None, None, None)
        x = self.norm(x)
        return self.proj(x).permute(0, 2, 1)

    @staticmethod
    def _load_spectral_prototypes(path, expected_num_classes):
        if not path:
            raise ValueError(
                "task_condition=True 时必须配置 model.spectral_prototype_path"
            )

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "prototypes" not in payload:
            raise ValueError(f"频谱原型文件格式无效，缺少 'prototypes': {path}")

        prototypes = torch.as_tensor(payload["prototypes"], dtype=torch.float32)
        if prototypes.ndim != 2:
            raise ValueError(
                f"频谱原型必须为 [num_classes, freq_dim]，实际为 {tuple(prototypes.shape)}"
            )
        if prototypes.shape[0] != expected_num_classes:
            raise ValueError(
                f"频谱原型类别数 {prototypes.shape[0]} 与配置 num_classes="
                f"{expected_num_classes} 不一致"
            )
        file_num_classes = payload.get("num_classes")
        if file_num_classes is not None and int(file_num_classes) != expected_num_classes:
            raise ValueError(
                f"原型文件 num_classes={file_num_classes} 与配置 "
                f"num_classes={expected_num_classes} 不一致"
            )
        if not torch.isfinite(prototypes).all():
            raise ValueError(f"频谱原型包含 NaN 或 Inf: {path}")
        return prototypes

    def _validate_labels(self, label):
        if label is None:
            raise ValueError("启用频谱原型条件时，forward 必须提供 label")
        if label.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"label 必须为整数张量，实际 dtype={label.dtype}")
        label = label.view(-1).long()
        if label.numel() == 0:
            raise ValueError("label 不能为空")
        if torch.any(label < 0) or torch.any(label >= self.num_classes):
            raise ValueError(
                f"label 必须位于 [0, {self.num_classes - 1}]，"
                f"实际范围 [{label.min().item()}, {label.max().item()}]"
            )
        return label


NoisePredictModel = TimeMixerNoisePredictor
