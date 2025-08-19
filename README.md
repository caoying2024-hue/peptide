# Evolutionary Information-Guided Deep Learning and Monte Carlo Simulation Annealing for MHCII Binding Peptide Design

## Introduction
It is a computational framework for short peptide design targeting MHCII molecules.  
It addresses the limitations of structure-based approaches for peptides lacking stable secondary structures by combining **evolutionary information** with **deep learning** and **Monte Carlo simulated annealing (MCSA)** strategies.  

---

## Features
- **Neural Network-based Design**
  - Uses amino acid frequency and residue pair joint frequency as learning signals.
  - CNN–Transformer architecture for modeling both local features and long-range dependencies of amino acid frequencies.
  - Supports multiple loss configurations to generate peptide sequences with different evolutionary constraints.
- **Monte Carlo Simulated Annealing (MCSA) Design**
  - Flexible initialization strategies (random or frequency-based).
  - Two types of mutation modes during annealing (random or probability-guided).
- **Output**
  - Designed peptide sequences (L1 and L2 from neural network; MC1 and MC2 from MCSA).

---

## Methods

### 1. Neural Network-based Method
Two modules are included:
- **train**: Train the neural network on peptide datasets to abtain trained weights.
- **test**: Use trained weights to predict amino acid probabilities.

Loss function modes:
- `loss2`: Uses **residue pair joint frequency** as the loss term → generates **L1 sequences**.
- `loss1 + 65*loss2`: Combines **amino acid frequency** and **residue pair joint frequency** → generates **L2 sequences**.

---

### 2. Monte Carlo Simulated Annealing Method
Two key settings are available:

- **Initialization mode**
  - `shuffle_columns`: Peptides are initialized according to the natural amino acid frequency distribution.
  - `shuffle_columns2`: Peptides are initialized completely at random.

- **Mutation mode**
  - `perturb_sequence`: Mutations are performed according to predefined amino acid frequency distributions.
  - `perturb_sequence2`: Mutations are performed randomly during annealing.

---

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.10
- NumPy

### Setup
https://github.com/caoying2024-hue/peptide.git
