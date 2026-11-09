"""Convolutional speech encoder with a projection head."""

import torch
import torch.nn as nn


class ConvFeatureExtractor(nn.Module):
    def __init__(self, channels=(512, 512, 512, 512, 512, 512, 512)):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
            layers.append(nn.GroupNorm(num_groups=out_channels // 64, num_channels=out_channels))
            layers.append(nn.GELU())
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, samples) -> (batch, time, channels)
        return self.layers(waveform.unsqueeze(1)).transpose(1, 2)


class ContrastiveSpeechEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768, projection_dim: int = 256):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, dropout=0.1),
            num_layers=6,
        )
        self.projector = nn.Sequential(
            nn.Linear(512, hidden_size), nn.GELU(), nn.Linear(hidden_size, projection_dim)
        )

    def forward(self, waveform: torch.Tensor):
        features = self.feature_extractor(waveform)
        encoded = self.encoder(features)
        pooled = encoded.mean(dim=1)
        return encoded, torch.nn.functional.normalize(self.projector(pooled), dim=-1)
