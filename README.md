# Hierarchical Concept-Based Explainable-by-Design Models

**Course Project (2025/2026)**
**Institution:** Politecnico di Torino
**Authors:** Luca Prato, Niccolò Sipala, Elziario Paduano

---

## Project Overview
This repository contains the implementation and analysis of hierarchical Concept Bottleneck Models (CBMs) for fine-grained bird classification on the CUB-200-2011 dataset.

The goal is to compare standard flat CBMs with hierarchical designs that incorporate:
- **Concept-level hierarchies**: group the 312 CUB attributes into anatomical part hierarchies (fine → mid → coarse).
- **Class-level hierarchies**: use WordNet-derived taxonomic structure to route classification through species, genus, and family levels.

This approach aims to improve interpretability while reducing severe taxonomic errors and producing richer multi-level explanations.

---

## Notebooks

- `baseline_cbm.ipynb`
  - Standard flat CBM baseline.
  - Uses ResNet-50 to predict 312 concept scores and then class logits for 200 bird species.
  - Evaluates top-1/top-5 accuracy, per-concept accuracy, and error severity metrics.

- `hier_mean_matrix.ipynb`
  - Hierarchical CBM using fixed aggregation from fine-grained attributes to mid-level and coarse-level concepts.
  - Includes anatomy-inspired structure over CUB parts and evaluates hierarchical classification performance.

- `hier_learnable_weights.ipynb`
  - Hierarchical CBM with learnable weights for concept aggregation.
  - Learns how much each mid-level and coarse-level group contributes to classification.

- `hier_wordnet_cbm.ipynb`
  - Class-level hierarchical CBM using WordNet taxonomy.
  - Builds a routed classification tree over families, genera, and species.
  - Adds local concept heads for internal taxonomy nodes and trains with teacher-forcing.

---
## Summary of Results

Our experimental evaluations on the CUB-200-2011 test split highlight a distinct accuracy-interpretability trade-off across the implemented architectures:

| Model Configuration | Top-1 Species Acc | Top-5 Species Acc | Concept Acc | Mean LCA Error Distance |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline CBM** (Flat) | **80.10%** | **95.56%** | 81.33% | 11.20 |
| **H-CBM (Mean Matrix)** | 63.67% | 91.08% | 68.32% | 11.42 |
| **H-CBM (Learnable Weights)** | 69.30% | 91.50% | **87.50%** | 11.08 |
| **WordNet CBM** (Taxonomic) | 78.13% | — | — | — |

### Key Findings:
1. **The Bottleneck Trade-off:** The flat Baseline CBM establishes an empirical upper bound for pure classification accuracy ($80.10\%$) due to its unconstrained access to all fine attributes.
2. **Learnable vs. Fixed Aggregation:** Fixed averaging (`hier_mean_matrix`) proves overly destructive, dropping performance significantly. Transitioning to learnable weights (`hier_learnable_weights`) recovers this discriminative gap, significantly boosting concept accuracy to **$87.50\%$** and reducing catastrophic cross-regional errors down to just **$3.20\%$**.
3. **Taxonomic Excellence:** The WordNet CBM (`hier_wordnet_cbm`) achieves highly competitive species accuracy ($78.13\%$) while successfully constraining classification mistakes to structurally close biological groups (Genus Acc: $84.64\%$, Family Acc: $87.40\%$), generating intuitive multi-level textual explanations.

---

## Dataset

This project uses the **CUB-200-2011** dataset.

Key dataset facts:
- 11,788 images of 200 bird species
- 312 annotated attributes
- Full train/test split from CUB
- Uses standard certainty filtering on attribute labels (`certainty >= 3`)

The notebooks download and extract the dataset automatically at runtime.

---

## ⚙️ Dependencies

The notebooks install their Python dependencies automatically when executed, but the main requirements are:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `Pillow`
- `tqdm`
- `nltk` (required for `hier_wordnet_cbm.ipynb`)

To install manually:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn Pillow tqdm nltk
```

---

## Usage

1. Open the notebook of interest in Jupyter, Colab, or another notebook environment.
2. Execute the first cells to install dependencies and set the random seed.
3. Run the dataset download and data-loading cells.
4. Execute the model training and evaluation cells.

Recommended order:
1. `baseline_cbm.ipynb`
2. `hier_mean_matrix.ipynb`
3. `hier_learnable_weights.ipynb`
4. `hier_wordnet_cbm.ipynb`

---

## Notes

- The project is notebook-driven; there are no separate Python scripts in the repository.
- Paths in the notebooks use `/content/data` and `/content/processed`, so the notebooks are ready for Google Colab. If running locally, adapt those paths as needed.
- The taxonomy model in `hier_wordnet_cbm.ipynb` relies on NLTK WordNet data and performs greedy top-down inference.

---

## 📚 Acknowledgements

- CUB-200-2011 dataset
- WordNet taxonomy via NLTK
- ResNet-50 backbone for concept and classification modeling
