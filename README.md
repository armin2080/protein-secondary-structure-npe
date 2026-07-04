# protein-secondary-structure-npe

Inference of protein secondary structure motifs using a two-state Hidden Markov Model and neural posterior estimation (BayesFlow).

## Project Overview

We simulate amino acid sequences using an HMM with two states (alpha-helix / other) and given transition/emission probabilities. The Forward-Backward algorithm computes true posterior state probabilities. A BayesFlow neural network is trained to approximate these posterior probabilities from amino acid sequences alone.

## Task Breakdown (Team of 4, 3 min each)

| Member | Role | Tasks |
|--------|------|-------|
| **1** | **Simulator & Data Lead** | Write HMM simulator (states, transitions, emission tables); run Forward-Backward to generate sequence-posterior pairs; preprocess and format human insulin 1A7F as real-world validation data |
| **2** | **Architecture Lead** | Design sequence-processing summary network (LSTM/Transformer); implement padding/masking for variable-length batches; configure the BayesFlow adapter and inference (coupling) network |
| **3** | **Optimization Lead** | Build the training pipeline; conduct hyperparameter tuning (optimizers, learning rate schedulers, regularization); output training diagnostics and loss trajectory curves |
| **4** | **Diagnostics & Inference Lead** | Execute Simulation-Based Calibration (SBC histograms, ECDFs); generate parameter recovery and z-score/contraction plots; run final inference on human insulin and compare to ground truth |

Slides are prepared together as a team.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `data_downloader.py` to download the Kaggle dataset to `dataset/`. The dataset is gitignored.