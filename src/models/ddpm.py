from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn

from src.models.frozen_eeg_encoder import FrozenEEGEncoder


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class DDPM(nn.Module):
    """条件 DDPM，用噪声预测网络从带噪 EEG 中恢复干净 EEG。"""

    def __init__(self, base_model, config, device, eps_scaler=1.0, conditional=True):
        super().__init__()
        self.device = device
        self.model = base_model
        self.config = config
        self.conditional = conditional
        self.eps_scaler = eps_scaler
        self.loss_func = nn.L1Loss(reduction="mean").to(device)
        train_config = config["train"]
        self.lambda_content = float(train_config.get("lambda_content", 0.0))
        self.lambda_style = float(train_config.get("lambda_style", 0.0))
        if self.lambda_content < 0 or self.lambda_style < 0:
            raise ValueError("lambda_content 和 lambda_style 不能小于 0")

        self.content_loss_func = nn.MSELoss(reduction="mean").to(device)
        self.style_loss_func = nn.MSELoss(reduction="mean").to(device)
        self.content_encoder = None
        self.encoder_ckpt_path = train_config.get("encoder_ckpt_path")
        if self.lambda_content > 0:
            if not self.encoder_ckpt_path:
                raise ValueError(
                    "lambda_content > 0 时必须配置 train.encoder_ckpt_path"
                )

        self.config_diff = config["diffusion"]
        self.sampling_scaler = [eps_scaler for _ in range(self.config_diff["q_num_steps"])]

        self.set_new_noise_schedule(device)

    def make_beta_schedule(self, schedule="linear", n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == "linear":
            return torch.linspace(start, end, n_timesteps)
        if schedule == "quadratic":
            return torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        if schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            return torch.sigmoid(betas) * (end - start) + start
        if schedule == "cosine":
            t = torch.linspace(0, n_timesteps, n_timesteps + 1)
            alphas_cumprod = torch.cos((t / n_timesteps + 0.008) * torch.pi / 2) ** 2
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return betas / betas.max() * (end - start) + start
        if schedule == "polynomial":
            return torch.linspace(start ** (1 / 3), end ** (1 / 3), n_timesteps) ** 3
        if schedule == "exponential":
            return torch.logspace(torch.log10(torch.tensor(start)), torch.log10(torch.tensor(end)), n_timesteps)
        raise ValueError(f"未知 diffusion schedule: {schedule}")

    def set_new_noise_schedule(self, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = self.make_beta_schedule(
            schedule=self.config_diff["schedule"],
            n_timesteps=self.config_diff["q_num_steps"],
            start=self.config_diff["beta_start"],
            end=self.config_diff["beta_end"],
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * (noise / self.eps_scaler)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised, condition_x=None, label=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        noise = self.predict_noise(x, t, condition_x=condition_x, label=label, noise_level=noise_level)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    def predict_noise(self, x, t, condition_x=None, label=None, noise_level=None):
        if noise_level is None:
            batch_size = x.shape[0]
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            return self.model(x, condition_x, label, noise_level)
        return self.model(x, None, label, noise_level)

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, condition_x=None, label=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            condition_x=condition_x,
            label=label,
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def ddim_sample(self, x, t, t_prev, condition_x=None, label=None, eta=0.0, clip_denoised=False):
        predicted_noise = self.predict_noise(x, t, condition_x=condition_x, label=label)
        x_start = self.predict_start_from_noise(x, t=t, noise=predicted_noise)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        alpha_t = self.alphas_cumprod[t]
        alpha_prev = torch.ones_like(alpha_t) if t_prev < 0 else self.alphas_cumprod[t_prev]
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        sigma = torch.clamp(sigma, min=0.0)

        effective_noise = predicted_noise / self.eps_scaler
        pred_direction = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0.0)) * effective_noise
        random_noise = sigma * torch.randn_like(x) if t_prev >= 0 and eta > 0 else torch.zeros_like(x)
        return torch.sqrt(alpha_prev) * x_start + pred_direction + random_noise

    @torch.no_grad()
    def p_sample_loop(self, x_in, label=None, continuous=False, sampler="ddpm", ddim_eta=0.0):
        sampler = sampler.lower()
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError(f"未知采样方法: {sampler}")

        device = self.betas.device
        q_steps = self.config_diff["q_num_steps"]
        if sampler == "ddpm":
            sample_steps = q_steps
            timesteps = np.arange(q_steps)
        else:
            sample_steps = int(self.config.get("eval", {}).get("ddim_num_steps", q_steps))
            sample_steps = max(1, min(sample_steps, q_steps))
            timesteps = np.linspace(0, q_steps - 1, sample_steps).astype(int)
        sample_inter = max(1, sample_steps // 10)
        if not self.conditional:
            cur_x = torch.randn_like(x_in).to(device) if torch.is_tensor(x_in) else torch.randn(x_in, device=device)
            ret_x = [cur_x]
            for i in reversed(range(sample_steps)):
                t = int(timesteps[i])
                t_prev = int(timesteps[i - 1]) if i > 0 else -1
                if sampler == "ddpm":
                    cur_x = self.p_sample(cur_x, t)
                else:
                    cur_x = self.ddim_sample(cur_x, t, t_prev, eta=ddim_eta)
                if i % sample_inter == 0:
                    ret_x.append(cur_x)
        else:
            condition_x = x_in
            cur_x = torch.randn(condition_x.shape, device=device)
            ret_x = [cur_x]
            for i in reversed(range(sample_steps)):
                t = int(timesteps[i])
                t_prev = int(timesteps[i - 1]) if i > 0 else -1
                if sampler == "ddpm":
                    cur_x = self.p_sample(cur_x, t, condition_x=condition_x, label=label)
                else:
                    cur_x = self.ddim_sample(
                        cur_x,
                        t,
                        t_prev,
                        condition_x=condition_x,
                        label=label,
                        eta=ddim_eta,
                    )
                if i % sample_inter == 0:
                    ret_x.append(cur_x)

        return ret_x if continuous else ret_x[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, shape=(1, 512), continuous=False, sampler="ddpm", ddim_eta=0.0):
        return self.p_sample_loop(
            (batch_size, shape[0], shape[1]),
            continuous=continuous,
            sampler=sampler,
            ddim_eta=ddim_eta,
        )

    @torch.no_grad()
    def denoising(self, x_in, label, continuous=False, sampler="ddpm", ddim_eta=0.0):
        return self.p_sample_loop(
            x_in,
            label,
            continuous=continuous,
            sampler=sampler,
            ddim_eta=ddim_eta,
        )

    def q_sample_loop(self, x_start, continuous=False):
        sample_inter = max(1, self.config_diff["q_num_steps"] // 10)
        ret_x = [x_start]
        cur_x = x_start
        for t in range(1, self.config_diff["q_num_steps"] + 1):
            batch_size = cur_x.shape[0]
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t - 1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=batch_size,
                )
            ).to(cur_x.device)
            noise = torch.randn_like(cur_x)
            cur_x = self.q_sample(
                x_start=cur_x,
                continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1),
                noise=noise,
            )
            if t % sample_inter == 0:
                ret_x.append(cur_x)
        return ret_x if continuous else ret_x[-1]

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if continuous_sqrt_alpha_cumprod.dim() == 1:
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(-1, 1, 1)
        elif continuous_sqrt_alpha_cumprod.dim() == 2:
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(-1, 1, 1)
        return continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise

    def sample_continuous_noise_level(self, batch_size, device):
        q_num_steps = self.config_diff["q_num_steps"]

        # 分层采样时间步，使一个 batch 尽量覆盖完整扩散时间范围。
        edges = torch.linspace(1, q_num_steps + 1, batch_size + 1, device=device)
        low_t = torch.floor(edges[:-1]).long().clamp(1, q_num_steps)
        high_t = (torch.ceil(edges[1:]).long() - 1).clamp(1, q_num_steps)
        high_t = torch.maximum(high_t, low_t)
        t = low_t + torch.floor(torch.rand(batch_size, device=device) * (high_t - low_t + 1).float()).long()
        t = t[torch.randperm(batch_size, device=device)]

        sqrt_alpha_table = torch.as_tensor(
            self.sqrt_alphas_cumprod_prev,
            dtype=torch.float32,
            device=device,
        )
        lower = sqrt_alpha_table[t - 1]
        upper = sqrt_alpha_table[t]

        # sqrt_alphas_cumprod_prev 通常随 t 递减，uniform 采样前显式校正上下界。
        interval_low = torch.minimum(lower, upper)
        interval_high = torch.maximum(lower, upper)
        continuous_sqrt_alpha_cumprod = interval_low + torch.rand(batch_size, device=device) * (
            interval_high - interval_low
        )
        return t, continuous_sqrt_alpha_cumprod.view(batch_size, 1)

    @staticmethod
    def normalized_power_spectrum(eeg):
        psd = torch.fft.rfft(eeg, dim=-1).abs().square()
        return psd / (psd.sum(dim=-1, keepdim=True) + 1e-8)

    def get_content_encoder(self):
        if self.content_encoder is None:
            self.content_encoder = FrozenEEGEncoder(
                self.encoder_ckpt_path
            ).to(self.device)
        return self.content_encoder

    def p_losses(self, clean_eeg, noisy_eeg, label, noise=None):
        x_start = clean_eeg
        batch_size = x_start.shape[0]
        t, continuous_sqrt_alpha_cumprod = self.sample_continuous_noise_level(batch_size, x_start.device)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod,
            noise=noise,
        )

        if not self.conditional:
            x_recon = self.model(x_noisy, None, label, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.model(x_noisy, noisy_eeg, label, continuous_sqrt_alpha_cumprod)

        loss_ddpm = self.loss_func(noise, x_recon)
        loss_content = loss_ddpm.new_zeros(())
        loss_style = loss_ddpm.new_zeros(())

        if self.lambda_content == 0 and self.lambda_style == 0:
            self.loss_components = {
                "ddpm": loss_ddpm.detach(),
                "content": loss_content,
                "style": loss_style,
            }
            return loss_ddpm

        alpha = continuous_sqrt_alpha_cumprod.view(-1, 1, 1)
        x0_pred = (x_noisy - (1 - alpha**2).sqrt() * x_recon) / alpha
        loss = loss_ddpm

        if self.lambda_content > 0:
            content_encoder = self.get_content_encoder()
            predicted_features = content_encoder(x0_pred)
            with torch.no_grad():
                noisy_features = content_encoder(noisy_eeg)
            loss_content = self.content_loss_func(
                predicted_features,
                noisy_features,
            )
            loss = loss + self.lambda_content * loss_content

        if self.lambda_style > 0:
            predicted_psd = self.normalized_power_spectrum(x0_pred)
            with torch.no_grad():
                clean_psd = self.normalized_power_spectrum(clean_eeg)
            loss_style = self.style_loss_func(predicted_psd, clean_psd)
            loss = loss + self.lambda_style * loss_style

        self.loss_components = {
            "ddpm": loss_ddpm.detach(),
            "content": loss_content.detach(),
            "style": loss_style.detach(),
        }
        return loss

    def forward(self, clean_eeg, noisy_eeg, label, *args, **kwargs):
        return self.p_losses(clean_eeg, noisy_eeg, label, *args, **kwargs)
