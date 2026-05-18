# REAP

REAP (Rank-guided Exploration for Automated enzyme reProgramming) is a closed-loop enzyme-engineering codebase for protein language model (PLM)-guided mutation design. It supports two related workflows:

1. validating RankReg against pointwise regression losses on fixed PLM embeddings;
2. applying PLM-RankReg to enzyme-mutation recommendation rounds, from zero-shot initialization to ensemble-based candidate ranking and wet-lab construction support.

## Directory layout

```text
REAP/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── src/reap/
│   ├── __init__.py
│   ├── assay.py              # assay replicate cleaning and merged training tables
│   ├── data.py               # embedding I/O, name alignment, splitting, and seeding
│   ├── embeddings.py         # ESM2 sequence-level embedding extraction
│   ├── esm_utils.py          # ESM model loading helpers
│   ├── losses.py             # RankReg and baseline regression losses
│   ├── models.py             # PLM-RankReg prediction heads
│   ├── mutation_design.py    # primer and mutant-plasmid design utilities
│   ├── selection.py          # ensemble prediction and UCB ranking
│   ├── sequence_utils.py     # sequence and mutation-notation utilities
│   ├── training.py           # training, evaluation, checkpointing, and baselines
│   └── zero_shot.py          # ESM-style zero-shot single-substitution scanning
├── scripts/
│   ├── 00_zero_shot_scan.py
│   ├── 01_extract_embeddings.py
│   ├── 02_clean_assay_data.py
│   ├── 03_prepare_training_data.py
│   ├── 04_train_plm_rankreg.py
│   ├── 05_train_ensemble.py
│   ├── 06_guide_mutations.py
│   ├── benchmark_loss_comparison.py
│   ├── benchmark_few_shot.py
│   ├── design_mut_primer.py
│   └── design_mut_plasmid.py
├── data/
│   ├── source/               # paper-associated source data after manuscript acceptance
│   ├── raw/                  # local FASTA, assay, codon-preference, and candidate files
│   ├── processed/            # cleaned assay tables and candidate sequence CSV files
│   └── embeddings/           # batch_*.npz PLM embedding files
├── checkpoints/
└── results/
```

## Data availability

This repository provides the code and expected data formats. We will release all source data associated with the paper under `./data/source` after the manuscript is accepted.

## Installation

Use Python 3.9 or newer. For GPU use, install the PyTorch build matching your CUDA driver first; then install the remaining dependencies and the local package:

```bash
git clone https://github.com/zyan-y/REAP.git
cd REAP
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Input formats

Embedding folders contain numerically ordered files named `batch_0.npz`, `batch_1.npz`, and so on.

Labeled training embeddings:

```text
batch_*.npz keys: X, y, n
```

Unlabeled candidate embeddings:

```text
batch_*.npz keys: X, n
```

`X` is a two-dimensional embedding matrix, `y` is a numeric activity/yield label, and `n` stores variant names.

Candidate sequence CSV files for embedding extraction use the first two or three columns:

```text
name, sequence, label_optional
```

Assay-cleaning input is an Excel file with:

```text
name, replicate1, replicate2, replicate3
```

The cleaned training table contains:

```text
name, yield
```

Codon-preference files for primer/plasmid design use one DNA codon per amino acid:

```text
A:GCT
C:TGT
D:GAT
```

Mutation strings use 1-based protein coordinates, for example `A123C`. Multi-mutants are written with hyphens, for example `A123C-G45D`. Plasmid design accepts multi-mutants; primer design expects one **single substitution** per row.

For plasmid-level design, `--cds_start` and `--cds_end` use zero-based, half-open nucleotide coordinates in the plasmid FASTA sequence.

## RankReg validation

The loss-comparison benchmark expects:

```text
data/embeddings/<dataset_name>/batch_*.npz
data/cv_folds/<dataset_name>.csv
```

The cross-validation CSV should contain the selected split column, typically `fold_random_5`.

```bash
python scripts/benchmark_loss_comparison.py \
  --embedding_folder data/embeddings \
  --cv_folder data/cv_folds \
  --cv_split fold_random_5 \
  --losses RankReg,MSE,Huber,L1 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --folds 0,1,2,3,4 \
  --standardize_y \
  --output_dir results/loss_comparison
```

Main outputs:

```text
results/loss_comparison/loss_comparison_raw.csv
results/loss_comparison/loss_comparison_summary_by_dataset.csv
results/loss_comparison/loss_comparison_summary_overall.csv
```

Optional few-shot benchmark:

```bash
python scripts/benchmark_few_shot.py \
  --embedding_folder data/embeddings \
  --losses RankReg,MSE,Huber,L1,EvolvePro \
  --train_sizes 50,100,200,400 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output_dir results/few_shot_comparison
```

## REAP workflow

### Step 1. Zero-shot single-substitution scan

```bash
python scripts/00_zero_shot_scan.py \
  --fasta data/raw/wt.fa \
  --output results/zero_shot/site_scan.csv
```

This writes single-substitution scores such as `A12G`.

### Step 2. Extract PLM embeddings

For one CSV file:

```bash
python scripts/01_extract_embeddings.py \
  --dms_file data/processed/candidates.csv \
  --embed_folder data/embeddings \
  --device cuda:0
```

For a folder of CSV files on multiple GPUs:

```bash
python scripts/01_extract_embeddings.py \
  --dms_folder data/processed \
  --embed_folder data/embeddings \
  --gpu_ids 0,1,2,3
```

### Step 3. Clean assay data

```bash
python scripts/02_clean_assay_data.py \
  --input_excel data/raw/round_1_assay.xlsx \
  --cleaned_dir data/processed/round_1
```

### Step 4. Align assay labels to embeddings

```bash
python scripts/03_prepare_training_data.py \
  --emb_dir data/embeddings/candidates \
  --table data/processed/round_1/data.xlsx \
  --out_dir data/embeddings/train_round_1 \
  --skip_missing
```

### Step 5. Train PLM-RankReg models

Single model:

```bash
python scripts/04_train_plm_rankreg.py \
  --embeddings_folder data/embeddings/train_round_1 \
  --output_dir checkpoints/round_1 \
  --model_type mlp \
  --alpha 0.8 \
  --margin 0.001
```

Ensemble for candidate ranking:

```bash
python scripts/05_train_ensemble.py \
  --embeddings_folder data/embeddings/train_round_1 \
  --output_dir checkpoints/round_1_ensemble \
  --ensemble_size 10 \
  --seeds 42,715,1388,2061,2734,3407,4080,4753,5426,6099
```

When `--standardize_y` is used, training checkpoints store the training-set mean and standard deviation. Ensemble prediction converts model outputs back to the original label scale before averaging.

### Step 6. Rank next-round candidates

```bash
python scripts/06_guide_mutations.py \
  --models_dir checkpoints/round_1_ensemble \
  --embeddings_folder data/embeddings/round_2_candidates \
  --output_dir results/round_2_selection \
  --lambda_sigma 0.5 \
  --top_n 93
```

The ranking score is:

```text
ucb_score = prediction_mean + lambda_sigma * prediction_std
```

Main outputs:

```text
results/round_2_selection/all_candidate_predictions.csv
results/round_2_selection/selected_candidates.xlsx
results/round_2_selection/ensemble_predictions.npy
```

In `selected_candidates.xlsx`, the `name` column contains the selected mutation names and appears first for compatibility with downstream design scripts.

### Step 7. Design primers for single substitutions

```bash
python scripts/design_mut_primer.py \
  --mutations_xlsx results/round_2_selection/selected_candidates.xlsx \
  --fasta data/raw/plasmid.fa \
  --cds_start 5066 \
  --codon_pref data/raw/codon_preference.txt \
  --save_path results/round_2_selection/design_primers.xlsx
```

The script reports forward/reverse primers, binding segments, mutation start coordinate, and estimated Tm values. The Excel file should contain single substitutions; split multi-mutants into individual construction steps before using this script.

### Step 8. Generate mutant plasmid FASTA files

```bash
python scripts/design_mut_plasmid.py \
  --mutations_xlsx results/round_2_selection/selected_candidates.xlsx \
  --plasmid_fasta data/raw/plasmid.fa \
  --cds_start 5066 \
  --cds_end 5516 \
  --enzyme P450 \
  --codon_pref data/raw/codon_preference.txt \
  --output_dir results/round_2_plasmids
```

This writes one FASTA file per requested mutant plasmid.

## Reproducibility

Training scripts expose random seeds and deterministic PyTorch settings. Benchmark scripts write raw per-run outputs and summary tables. Candidate-selection outputs include `prediction_mean`, `prediction_std`, `lambda_sigma`, and `ucb_score`.

## License

This project is licensed under AGPL-3.0-only. The full license text is provided in `LICENSE`; third-party dependencies are distributed under their own licenses.

## Citation

If you use REAP or RankReg in academic work, please cite the associated paper when it becomes available. A BibTeX entry will be added after publication.
