from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def snr_dirname(snr_db):
    value = float(snr_db)
    if value.is_integer():
        value = int(value)
    text = str(value).replace("-", "m").replace(".", "p")
    return f"snr_{text}db"


class SNRBenchmark:
    """In-memory view of the unified rest/motion fixed-SNR benchmark."""

    SPLITS = ("rest", "motion")

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"SNR benchmark 不存在: {self.path}")
        with np.load(self.path, allow_pickle=False) as data:
            self.snr_db = np.asarray(data["snr_db"], dtype=np.float32)
            self._arrays = {}
            for split in self.SPLITS:
                self._arrays[split] = {
                    "clean_eeg": np.asarray(
                        data[f"{split}_clean_eeg"], dtype=np.float32
                    ),
                    "noisy_eeg": np.asarray(
                        data[f"{split}_noisy_eeg"], dtype=np.float32
                    ),
                    "label": np.asarray(data[f"{split}_label"], dtype=np.int64),
                }
        self._validate()

    def _validate(self):
        if self.snr_db.ndim != 1 or len(self.snr_db) == 0:
            raise ValueError("snr_db 必须是一维非空数组")
        for split, arrays in self._arrays.items():
            clean = arrays["clean_eeg"]
            noisy = arrays["noisy_eeg"]
            label = arrays["label"]
            if clean.ndim != 3 or clean.shape[1] != 1:
                raise ValueError(
                    f"{split}_clean_eeg 必须为 [N,1,T]，实际为 {clean.shape}"
                )
            expected = (len(self.snr_db), *clean.shape)
            if noisy.shape != expected:
                raise ValueError(
                    f"{split}_noisy_eeg 应为 {expected}，实际为 {noisy.shape}"
                )
            if label.shape != (len(clean),):
                raise ValueError(
                    f"{split}_label 应为 {(len(clean),)}，实际为 {label.shape}"
                )
            if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
                raise ValueError(f"{split} 数据包含 NaN 或 Inf")

    def dataset(self, split, snr_index):
        if split not in self.SPLITS:
            raise ValueError(f"split 必须是 {self.SPLITS}，实际为 {split}")
        if not 0 <= snr_index < len(self.snr_db):
            raise IndexError(f"snr_index 越界: {snr_index}")
        arrays = self._arrays[split]
        return SNRBenchmarkDataset(
            noisy_eeg=arrays["noisy_eeg"][snr_index],
            clean_eeg=arrays["clean_eeg"],
            label=arrays["label"],
            snr_db=float(self.snr_db[snr_index]),
        )

    def write_slice(self, path, split, snr_index):
        dataset = self.dataset(split, snr_index)
        np.save(
            path,
            {
                "noisy_eeg": dataset.noisy_eeg,
                "clean_eeg": dataset.clean_eeg,
                "label": dataset.label,
            },
        )


class SNRBenchmarkDataset(Dataset):
    def __init__(self, noisy_eeg, clean_eeg, label, snr_db):
        self.noisy_eeg = noisy_eeg
        self.clean_eeg = clean_eeg
        self.label = label
        self.snr_db = snr_db

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {
            "noisy_eeg": torch.as_tensor(
                self.noisy_eeg[index], dtype=torch.float32
            ),
            "clean_eeg": torch.as_tensor(
                self.clean_eeg[index], dtype=torch.float32
            ),
            "label": torch.as_tensor(self.label[index], dtype=torch.long),
            "snr_db": torch.tensor(self.snr_db, dtype=torch.float32),
        }
