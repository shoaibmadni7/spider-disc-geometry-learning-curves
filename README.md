# Disc geometry versus deep image features on SPIDER: released code

Code for the robustness analyses (Section 5.8) of "Data Requirements of Interpretable Disc Geometry Versus Deep Image
Features for Lumbar Degeneration Grading on MRI: A Learning-Curve Study".

## Layout
The two data folders are stored as archives. After cloning or downloading, unzip `original_outputs.zip` and
`revision_experiments.zip` in the repository root; the scripts expect the folders `original_outputs/` and
`revision_experiments/`. The Colab notebooks (`SPIDER_Notebook*.ipynb`) are in the repository root.

## Data
Download `masks.zip`, `overview.csv` and `radiological_gradings.csv` from the SPIDER dataset
(Zenodo, doi:10.5281/zenodo.10159290, CC BY 4.0). Unzip the masks to `../data/masks/` and set `U` in `common.py`
to the folder holding the two CSV files.

## Run
    pip install -r requirements.txt
    python pert.py      # feature extraction from reference and perturbed masks (P_*.csv)
    python run.py       # experiments E0-E5, writes results.json
    python e6.py        # fold-wise exponent selection inside learning-curve subsets

## Files
- `extract.py`  geometric feature extractor (rotated disc frame, footprint restriction) and mask perturbation
- `common.py`   cohort construction, plausibility rule, per-level normalisation, classifier, fold generation
- `exp.py`      E0 metrics and calibration; E1 fold-nested exponents and level-6 sensitivity; E2 learning curves
                (5 partitions x 5 orderings); E3 mask perturbation; E4 leave-one-scanner-out; E5 single-sequence analysis
- `cohort_table.csv`  patient and level identifiers, plausibility flag, inclusion flag, features and targets
- `results.json`      mask-perturbation and single-sequence results (released extractor)
- `orig.py`, `orig_results.json`  analyses on the original features and predictions

## Original features and predictions
`original_outputs/` holds the feature table of the paper (`spider_primary_cohort.csv`, `spider_features.csv`), the DINOv2
embeddings, the crop index (row and sequence of each crop) and the saved out-of-fold predictions of all models
(`nb3_oof_rep.npy`, `nb3_cnn/`, `nb4_learning_curve/`). `python orig.py` recomputes from them Tables 3, 4, 7, 8 and 9 and
the exponent, level-6 and T2-crop analyses of Section 5.8 (`orig_results.json`).

## Revision experiments (GPU)
`SPIDER_Notebook8B_revision_experiments.ipynb` (Colab, Tesla T4, PyTorch 2.14.1) produced `revision_experiments/`:
learning curves on three partitions x three orderings under three CNN training protocols (`lc/`), three CNN seeds per
held-out scanner (`loso/`) and the augmented EfficientNet-B0 and ConvNeXt-Tiny comparators (`sensitivity/`); per-fold
update counts and seeds are stored inside each file. `python b8.py` recomputes Table 7, Table 8 and the comparator results
(`b8_results.json`).

## Seeds
Repeated cross-validation: `StratifiedGroupKFold(5, shuffle=True, random_state=rep)`, rep = 0..4.
Learning curves: outer partition `random_state=100+part`; patient ordering `default_rng(10000*ordering+100*part+fold)`.
Bootstrap: `default_rng(0..3)` as set in `exp.py`; patients are resampled with replacement.

## Relation to the original implementation
This is a re-implementation of the extractor from the Methods text. It yields 1,088 discs at levels 1-5 and descriptor
AUCs of 0.886, 0.759, 0.839 and 0.774 (narrowing, bulging, Pfirrmann >= 4, Modic), within 0.016 of Table 3 of the paper.
The image-arm experiments (DINOv2, EfficientNet-B0, fusion) are the `SPIDER_Notebook*.ipynb` files.
