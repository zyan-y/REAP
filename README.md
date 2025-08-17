# REAP

This repository contains the code of REAP (Rank-guided Exploration for Autonomous enzyme reProgramming), a closed-loop platform that integrates machine learning–guided design with robotic experimentation.

## Overview

Enzyme reprogramming underpins new catalysts and therapeutics but is hard to scale with conventional engineering. **REAP** integrates a **rank-guided predictor** trained with  **RankReg** —a hybrid loss that jointly optimizes ranking fidelity and quantitative accuracy—into an autonomous loop. The model learns **in-loop** from high-quality experimental feedback, enabling efficient navigation of complex fitness landscapes.

## What's in this repository

* **RankReg loss**: reference implementation of the RankReg hybrid loss.
* **PLM-RankReg model**: training and inference code for the rank-guided predictor.
* **REAP loop**: scripts/utilities to run model-guided design, batch prediction, and interface with automated experiments in a closed loop.
* **Experimental data**: raw datasets used to train/evaluate the models and to drive the in-loop updates.

## Getting started

1. **Clone & install**

   ```
   git clone https://github.com/zyan-y/REAP.git
   cd REAP
   pip install -r requirements.txt
   ```
2. **Make zero-shot predictions for Initialization**

   Apply the ESM2 model to the target enzyme and generate zero-shot predictions for all single-point mutants.
3. **Predict & score**

   Training code learns both ranking consistency and numeric accuracy. See the training scripts for flags controlling learning rate, optimizer, and RankReg coefficients. Use the provided inference utilities to generate ranked candidate sets and quantitative scores for validation.
4. **Run a REAP iteration**

   The loop modules orchestrate: candidate design → prediction/ranking → experimental queue generation → data ingestion → model update.

## Reproducibility

* Seeds and deterministic flags are exposed in training/inference scripts.
* Logs and metrics (including ranking metrics) are written to the configured output directory.

## License

This project is licensed under  **AGPL-3.0-only** . See `LICENSE` for details.

### Third-party components

This repository may include or depend on third-party software distributed under their respective licenses (e.g., components under  **MIT** ). Their notices are retained in `THIRD_PARTY_NOTICES` or within source headers.

## Patents

A **patent application** covering aspects of REAP has been filed. Use of this code is permitted only under the terms of the AGPL-3.0 license; no additional patent rights are granted beyond those provided by that license.

## Citation

If you use REAP or RankReg in academic work, please cite the associated paper (when available). A BibTeX entry will be provided in this repository.
