import os
import glob

import numpy as np
from torch.utils.data import Dataset


class BirdclefDataset(Dataset):
    def __init__(self, data_path: str, set_type: str, **kwargs):
        self.data_path = data_path
        self.class_names = sorted(
            [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        )
        if set_type == "train":
            self.label2id = {label: idx for idx, label in enumerate(self.class_names)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
        else:
            self.label2id = kwargs["label2id"]
            self.id2label = kwargs["id2label"]

        self.samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(data_path, class_name)
            clip_paths = glob.glob(os.path.join(class_dir, "*.npy"))
            for clip_path in clip_paths:
                self.samples.append({"clip_path": clip_path, "label": self.label2id[class_name]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        spec = np.load(sample["clip_path"]).astype(np.float32)  # (n_mels, time)
        label = sample["label"]

        # Add channel dimension
        spec = spec[np.newaxis, :, :]  # (1, n_mels, time)

        return spec, label
