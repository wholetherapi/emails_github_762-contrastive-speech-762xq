# Contrastive Pretraining for Low-Resource Speech Recognition

This repository contains the implementation for the paper
**"Contrastive Pretraining for Low-Resource Speech Recognition"**
(Interspeech 2025).

The release includes the contrastive pretraining objective, the wav2vec-style
encoder and the fine-tuning scripts for low-resource ASR. Trained checkpoints
are released in `checkpoints/` releases of this repository.

## Quick start

```bash
pip install -r requirements.txt
python -m src.contrastive_speech.train --manifest data/train.tsv --epochs 30
```

## Citation

```bibtex
@inproceedings{morales2025contrastive,
  title={Contrastive Pretraining for Low-Resource Speech Recognition},
  author={Morales, Patrick and Lindqvist, Sofia and others},
  booktitle={Proceedings of Interspeech 2025},
  year={2025}
}
```
