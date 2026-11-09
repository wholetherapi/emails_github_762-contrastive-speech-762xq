"""Fine-tuning entry point for low-resource ASR."""

import argparse
import torch
from torch.utils.data import DataLoader

from .model import ContrastiveSpeechEncoder
from .losses import contrastive_pretraining_loss


def train(manifest: str, epochs: int, batch_size: int, lr: float):
    dataset = load_manifest(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ContrastiveSpeechEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for waveforms, _ in loader:
            views = [augment(waveforms) for _ in range(2)]
            _, proj_a = model(views[0])
            _, proj_b = model(views[1])
            loss = contrastive_pretraining_loss([proj_a, proj_b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch} loss={loss.item():.4f}")


def load_manifest(path: str):
    import soundfile as sf
    import torch

    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            audio_path, transcript = line.strip().split("\t")
            waveform, _ = sf.read(audio_path, dtype="float32")
            items.append((torch.tensor(waveform), transcript))
    return items


def augment(waveform: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(waveform) * 0.005
    return waveform + noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(args.manifest, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
