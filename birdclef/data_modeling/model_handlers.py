import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


class SpecAugment(nn.Module):
    def __init__(self, time_mask=30, freq_mask=13):
        super().__init__()
        self.time_mask = time_mask
        self.freq_mask = freq_mask

    def forward(self, x):
        if not self.training:
            return x

        for i in range(x.size(0)):
            t = torch.randint(0, self.time_mask, (1,)).item()
            f = torch.randint(0, self.freq_mask, (1,)).item()

            t0 = torch.randint(0, max(1, x.size(3) - t), (1,)).item()
            f0 = torch.randint(0, max(1, x.size(2) - f), (1,)).item()

            x[i, 0, f0: f0 + f, :] = 0
            x[i, 0, :, t0: t0 + t] = 0

        return x


class BirdCLEFCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        self.dropout = nn.Dropout2d(0.3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
