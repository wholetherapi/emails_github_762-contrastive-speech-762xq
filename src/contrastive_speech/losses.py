"""InfoNCE losses for contrastive speech pretraining."""

import torch
import torch.nn.functional as F


def nt_xent_loss(anchor: torch.Tensor, positive: torch.Tensor,
                 temperature: float = 0.1) -> torch.Tensor:
    """Normalised temperature-scaled cross entropy over a batch."""
    batch_size = anchor.size(0)
    logits = torch.matmul(anchor, positive.t()) / temperature
    labels = torch.arange(batch_size, device=anchor.device)
    loss = F.cross_entropy(logits, labels)
    loss = loss + F.cross_entropy(logits.t(), labels)
    return 0.5 * loss


def contrastive_pretraining_loss(views: list, temperature: float = 0.1):
    """Average InfoNCE over all pairs of augmented views of the same utterance."""
    total, count = 0.0, 0
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            total = total + nt_xent_loss(views[i], views[j], temperature)
            count += 1
    return total / max(count, 1)
