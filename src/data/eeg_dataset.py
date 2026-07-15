import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """从预处理好的 npy 字典文件加载 EEG 去噪样本。"""

    def __init__(self, npy_file, transform=None):
        data = np.load(npy_file, allow_pickle=True).item()
        self.noisy_eeg = data["noisy_eeg"]
        self.clean_eeg = data["clean_eeg"]
        self.label = data["label"]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_eeg)

    def __getitem__(self, idx):
        noisy = torch.tensor(self.noisy_eeg[idx], dtype=torch.float32)
        clean = torch.tensor(self.clean_eeg[idx], dtype=torch.float32)
        label = torch.tensor(self.label[idx], dtype=torch.long)

        if self.transform:
            noisy, clean = self.transform(noisy, clean)

        return {
            "noisy_eeg": noisy,
            "clean_eeg": clean,
            "label": label,
        }
