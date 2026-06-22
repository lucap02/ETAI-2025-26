# Hierarchical Concept-Based Explainable-by-Design Models

**Course Project (2025/2026)**
**Institution:** Politecnico di Torino
**Authors:** Luca Prato, Niccolò Sipala, Elziario Paduano

---

## 📌 Project Overview
This repository contains the implementation and analysis of hierarchical Concept Bottleneck Models (CBMs) for fine-grained bird classification on the CUB-200-2011 dataset.

The goal is to compare standard flat CBMs with hierarchical designs that incorporate:
- **Concept-level hierarchies**: group the 312 CUB attributes into anatomical part hierarchies (fine → mid → coarse).
- **Class-level hierarchies**: use WordNet-derived taxonomic structure to route classification through species, genus, and family levels.

This approach aims to improve interpretability while reducing severe taxonomic errors and producing richer multi-level explanations.

---

## 🧠 Notebooks

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

## 📦 Dataset

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

## ▶️ Usage

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

## 📌 Notes

- The project is notebook-driven; there are no separate Python scripts in the repository.
- Paths in the notebooks use `/content/data` and `/content/processed`, so the notebooks are ready for Google Colab. If running locally, adapt those paths as needed.
- The taxonomy model in `hier_wordnet_cbm.ipynb` relies on NLTK WordNet data and performs greedy top-down inference.

---

## 📚 Acknowledgements

- CUB-200-2011 dataset
- WordNet taxonomy via NLTK
- ResNet-50 backbone for concept and classification modeling
