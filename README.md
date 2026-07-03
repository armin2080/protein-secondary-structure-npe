# protein-secondary-structure-npe

Inference of protein secondary structure motifs using a two-state Hidden Markov Model and neural posterior estimation (BayesFlow).

## Project Overview

We simulate amino acid sequences using an HMM with two states (alpha-helix / other) and given transition/emission probabilities. The Forward-Backward algorithm computes true posterior state probabilities. A BayesFlow neural network is trained to approximate these posterior probabilities from amino acid sequences alone.

## Suggested task breakdown

| Member | Tasks |
|--------|-------|
| **1** | HMM Simulator + Forward-Backward + Analyze real Kaggle dataset (compute empirical emission/transition frequencies, compare to given HMM parameters) |
| **2** | BayesFlow architecture & data pipeline (summary network + coupling network + dataloader for variable-length sequences) |
| **3** | BayesFlow training loop & hyperparameters (train model, track loss, convergence diagnostics, save weights) |
| **4** | Synthetic evaluation (metrics, calibration plots, predicted vs true posteriors) + Real insulin validation (test on PDB: 1A7F) |

Slides are prepared together as a team.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `data_downloader.py` to download the Kaggle dataset to `dataset/`. The dataset is gitignored.