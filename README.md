# Universal MLIPs for ferroelectric oxides: a benchmark against a consistent DFT reference

Data and code for: O. Azaroual, *How accurately do universal machine-learning interatomic potentials describe ferroelectric energy landscapes? A benchmark of 14 models against a consistent DFT reference, with finite-temperature and fine-tuning tests* (submitted).

## Contents
- `data/mlip/` – predictions of 14 universal MLIPs (MACE ×7, CHGNet, SevenNet ×2, MatterSim ×2, ORB-v3 ×2):
  - `results_step2_mace_all.csv`, `results_step4_<family>.csv`: cubic, P4mm, Amm2, R3m phases of BaTiO3, PbTiO3, KNbO3, SrTiO3, KTaO3 (energies, lattice, space group, phonons at Γ/X/M/R of the cubic phase);
  - `results_step5_<family>.csv`: P2_1/c, Pbca, Pca2_1, P4_2/nmc, Fm-3m polymorphs of HfO2 and ZrO2;
  - `results_step5_mp_reference.csv`: Materials Project IDs and energies of the starting structures.
- `data/dft/` – Quantum ESPRESSO DFT-PBE reference (GBRV pseudopotentials; PseudoDojo check for BaTiO3/PbTiO3) and the starting structures of the oxides (`step5_structures.extxyz`).
- `data/md/` – NPT molecular dynamics (4×4×4 supercells, stepwise heating): ORB-v3 (MPA) for BaTiO3 (two runs 50–600 K, one 650–1000 K) and PbTiO3 (200–1500 K); MACE-MPA-0 and the fine-tuned MACE-MPA-0 for BaTiO3 (50–900 K) and PbTiO3 (200–400 K). Columns: average Ti off-centring (Px, Py, Pz, Å), lattice parameters, temperature.
- `data/finetune/` – fine-tuning of MACE-MPA-0: `dft_train_all.extxyz` (209 DFT-PBE configurations of BaTiO3 and PbTiO3 with REF_energy, REF_forces, REF_stress), validation errors (`ft_errors.csv`), static benchmark of the original and fine-tuned models (`ft_static.csv`), settings (`ft_summary.json`) and an example training log. The fine-tuned model (`mace_ft_BTO_PTO.model`, 79 MB, SHA-256 3d2d9fb391a3d60de7538f58be68f0cbebcad037b223dc4f3cbed0eeb820cff1) is attached to the GitHub release v1.0 of this repository (load with `mace.calculators.MACECalculator`).
- `notebooks/` – the Google Colab (MLIP) and Kaggle (DFT) notebooks used to produce all data.
- `analysis/analysis.py` – builds the benchmark table, scores and figures 1–5; `analysis/analysis_ft_md.py` – fine-tuning and MD analysis, figures 6–7 (run from a folder containing the data files).
- `figures/` – figures of the paper.

## Requirements
MLIP notebooks: Google Colab with a T4 GPU (packages installed in the notebooks). Molecular dynamics and fine-tuning (steps 7–9): Kaggle GPU sessions. DFT notebooks: Kaggle CPU sessions (Quantum ESPRESSO from conda-forge). Analysis: Python 3 with pandas, numpy, scipy, matplotlib.

## License
Data: CC BY 4.0. Code: MIT.
