import csv
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.lib.format import open_memmap

from src.utils import evaluation_root, sampling_variant


@torch.no_grad()
def evaluate_ddpm(model, test_loader, config, logger, eeg_type):
    eval_cfg = config["eval"]
    sampler = eval_cfg.get("sampler", "ddpm")
    if sampler == "ddpm":
        sample_steps = config["diffusion"]["q_num_steps"]
    else:
        sample_steps = eval_cfg.get("ddim_num_steps", config["diffusion"]["q_num_steps"])
    ddim_eta = eval_cfg.get("ddim_eta", 0.0)
    logger.info(f"采样方法: {sampler}, 采样步数: {sample_steps}, ddim_eta: {ddim_eta}")

    dataset_root = evaluation_root(config) / eeg_type / sampling_variant(config)
    figure_dir = dataset_root / "figures"
    dataset_root.mkdir(parents=True, exist_ok=True)
    if eval_cfg["save_figures"]:
        figure_dir.mkdir(parents=True, exist_ok=True)

    save_denoised = eval_cfg.get("save_denoised", True)
    prediction_writer = PredictionWriter(dataset_root, len(test_loader.dataset)) if save_denoised else None
    metric_sums = {"cc": 0.0, "rrmse": 0.0, "snr": 0.0, "rrmse_freq": 0.0}
    num_samples = 0
    num_batches = 0

    model.eval()
    metric_path = dataset_root / "segment_metrics.csv"
    metric_fields = ["eeg_type", "batch", "sample_index", "cc", "rrmse", "snr", "rrmse_freq"]
    with open(metric_path, "w", newline="", encoding="utf-8") as metric_file:
        metric_writer = csv.DictWriter(metric_file, fieldnames=metric_fields)
        metric_writer.writeheader()

        for batch_no, batch in enumerate(test_loader, start=1):
            clean_batch = batch["clean_eeg"].to(model.device)
            noisy_batch = batch["noisy_eeg"].to(model.device)
            label = batch["label"].to(model.device)
            denoised = model.denoising(noisy_batch, label, sampler=sampler, ddim_eta=ddim_eta)

            cc_values = cal_acc_per_sample(denoised, clean_batch)
            rrmse_values = cal_rrmse_per_sample(denoised, clean_batch)
            snr_values = cal_snr_per_sample(denoised, clean_batch)
            rrmse_freq_values = cal_rrmse_freq_per_sample(denoised, clean_batch)

            batch_cc = torch.mean(cc_values)
            batch_rrmse = torch.mean(rrmse_values)
            batch_snr = float(np.mean(snr_values))
            batch_rrmse_freq = torch.mean(rrmse_freq_values)

            logger.info(
                f"{eeg_type} batch={batch_no} "
                f"cc={batch_cc.item()} rrmse={batch_rrmse.item()} snr={batch_snr} "
                f"rrmse_freq={batch_rrmse_freq.item()}"
            )

            batch_size = noisy_batch.shape[0]
            sample_start = num_samples
            sample_end = sample_start + batch_size
            sample_indices = np.arange(sample_start, sample_end)
            cc_np = cc_values.detach().cpu().numpy()
            rrmse_np = rrmse_values.detach().cpu().numpy()
            rrmse_freq_np = rrmse_freq_values.detach().cpu().numpy()

            for local_idx in range(batch_size):
                metric_writer.writerow(
                    {
                        "eeg_type": eeg_type,
                        "batch": batch_no,
                        "sample_index": int(sample_indices[local_idx]),
                        "cc": float(cc_np[local_idx]),
                        "rrmse": float(rrmse_np[local_idx]),
                        "snr": float(snr_values[local_idx]),
                        "rrmse_freq": float(rrmse_freq_np[local_idx]),
                    }
                )

            metric_sums["cc"] += float(np.sum(cc_np))
            metric_sums["rrmse"] += float(np.sum(rrmse_np))
            metric_sums["snr"] += float(np.sum(snr_values))
            metric_sums["rrmse_freq"] += float(np.sum(rrmse_freq_np))
            num_samples += batch_size
            num_batches = batch_no

            if prediction_writer is not None:
                prediction_writer.write(
                    noisy_batch.detach().cpu().numpy(),
                    clean_batch.detach().cpu().numpy(),
                    denoised.detach().cpu().numpy(),
                    label.detach().cpu().numpy(),
                    sample_indices,
                )

            if eval_cfg["save_figures"] and batch_no % eval_cfg["figure_interval"] == 0:
                save_signal_figure(noisy_batch, clean_batch, denoised, figure_dir, batch_no, eeg_type)

    if num_samples == 0:
        logger.warning(f"{eeg_type} 测试集为空，未生成指标和去噪结果")
        return {}

    result = {
        "eeg_type": eeg_type,
        "sampler": sampler,
        "sample_steps": int(sample_steps),
        "ddim_eta": float(ddim_eta),
        "num_batches": int(num_batches),
        "num_samples": int(num_samples),
        "cc": metric_sums["cc"] / num_samples,
        "rrmse": metric_sums["rrmse"] / num_samples,
        "snr": metric_sums["snr"] / num_samples,
        "rrmse_freq": metric_sums["rrmse_freq"] / num_samples,
    }
    if prediction_writer is not None:
        prediction_writer.finalize(dataset_root / "predictions.npz")
    write_summary(dataset_root, result)
    logger.info(f"{eeg_type} 平均指标: {result}")
    logger.info(f"{eeg_type} 去噪结果保存目录: {dataset_root}")
    return result


def cal_acc(predict, truth):
    return torch.mean(cal_acc_per_sample(predict, truth))


def cal_acc_per_sample(predict, truth):
    vy_pred = predict - torch.mean(predict, dim=-1, keepdim=True)
    vy_truth = truth - torch.mean(truth, dim=-1, keepdim=True)
    cc = torch.sum(vy_pred * vy_truth, dim=-1) / (
        torch.sqrt(torch.sum(vy_pred**2, dim=-1)) * torch.sqrt(torch.sum(vy_truth**2, dim=-1)) + 1e-8
    )
    return torch.mean(cc, dim=1)


def cal_rrmse(predict, truth):
    return torch.mean(cal_rrmse_per_sample(predict, truth))


def cal_rrmse_per_sample(predict, truth):
    mse = torch.mean((predict - truth) ** 2, dim=-1)
    signal_power = torch.mean(truth**2, dim=-1)
    return torch.mean(torch.sqrt(mse) / (torch.sqrt(signal_power) + 1e-8), dim=1)


def cal_snr(predict, truth):
    return float(np.mean(cal_snr_per_sample(predict, truth)))


def cal_snr_per_sample(predict, truth):
    predict = predict.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy()
    signal_power = np.sum(np.square(truth), axis=-1)
    noise_power = np.sum(np.square(predict - truth), axis=-1)
    noise_power = np.where(noise_power == 0, 1e-8, noise_power)
    return np.mean(10 * np.log10(signal_power / noise_power), axis=1)


def cal_rrmse_freq(predict, truth):
    return torch.mean(cal_rrmse_freq_per_sample(predict, truth))


def cal_rrmse_freq_per_sample(predict, truth):
    predict_mag = torch.abs(torch.fft.fft(predict, dim=-1))
    truth_mag = torch.abs(torch.fft.fft(truth, dim=-1))
    mse = torch.mean((predict_mag - truth_mag) ** 2, dim=-1)
    norm = torch.mean(truth_mag**2, dim=-1)
    return torch.mean(torch.sqrt(mse) / (torch.sqrt(norm) + 1e-8), dim=1)


def save_signal_figure(noisy_batch, clean_batch, denoised_batch, figure_dir, batch_no, eeg_type):
    sample_idx = 0
    noisy_signal = noisy_batch[sample_idx, 0].detach().cpu().numpy()
    clean_signal = clean_batch[sample_idx, 0].detach().cpu().numpy()
    denoised_signal = denoised_batch[sample_idx, 0].detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(noisy_signal, label="Noisy Signal", alpha=0.7)
    plt.plot(clean_signal, label="EEG Signal", alpha=0.7)
    plt.plot(denoised_signal, label="Denoised Signal", alpha=0.9)
    plt.legend()
    plt.title(f"EEG Denoising Batch:{batch_no} Sample:{sample_idx}")
    plt.savefig(figure_dir / f"batch_{batch_no}_sample_{sample_idx}_{eeg_type}.png", dpi=300)
    plt.close()


def write_summary(metric_dir, summary):
    with open(metric_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


class PredictionWriter:
    def __init__(self, dataset_root, total_samples):
        self.total_samples = int(total_samples)
        self.offset = 0
        self.tmp_dir = dataset_root / ".tmp_predictions"
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.arrays = {}

    def write(self, noisy, clean, denoised, label, sample_index):
        batch_size = noisy.shape[0]
        self._ensure_arrays(noisy, clean, denoised, label)
        start = self.offset
        end = start + batch_size
        self.arrays["noisy_eeg"][start:end] = noisy
        self.arrays["clean_eeg"][start:end] = clean
        self.arrays["denoised_eeg"][start:end] = denoised
        self.arrays["label"][start:end] = label
        self.arrays["sample_index"][start:end] = sample_index
        self.offset = end

    def finalize(self, path):
        for array in self.arrays.values():
            array.flush()
        np.savez_compressed(
            path,
            noisy_eeg=self.arrays["noisy_eeg"][: self.offset],
            clean_eeg=self.arrays["clean_eeg"][: self.offset],
            denoised_eeg=self.arrays["denoised_eeg"][: self.offset],
            label=self.arrays["label"][: self.offset],
            sample_index=self.arrays["sample_index"][: self.offset],
        )
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _ensure_arrays(self, noisy, clean, denoised, label):
        if self.arrays:
            return
        self.arrays = {
            "noisy_eeg": self._memmap("noisy_eeg.npy", noisy.dtype, (self.total_samples, *noisy.shape[1:])),
            "clean_eeg": self._memmap("clean_eeg.npy", clean.dtype, (self.total_samples, *clean.shape[1:])),
            "denoised_eeg": self._memmap(
                "denoised_eeg.npy",
                denoised.dtype,
                (self.total_samples, *denoised.shape[1:]),
            ),
            "label": self._memmap("label.npy", label.dtype, (self.total_samples, *label.shape[1:])),
            "sample_index": self._memmap("sample_index.npy", np.int64, (self.total_samples,)),
        }

    def _memmap(self, name, dtype, shape):
        return open_memmap(self.tmp_dir / name, mode="w+", dtype=dtype, shape=shape)
