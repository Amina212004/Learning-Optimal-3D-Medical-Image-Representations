# Reproduction and Exploration of Axial-Centric Cross-Plane Attention for 3D Medical Image Classification

<p align="center">
  **Medical AI · 3D Medical Imaging · Vision Foundation Models · Multi-View Learning · Few-Shot Learning**
</p>

<p align="center">
  A research implementation of a multi-plane architecture for 3D medical image classification, with an additional investigation of few-shot learning on the resulting representations.
</p>

---

## 1. Project Overview

This project focuses on the implementation and experimental exploration of the research paper:

**[Axial-Centric Cross-Plane Attention for 3D Medical Image Classification](https://arxiv.org/abs/2602.21636)**

**Authors:** Doyoung Park, Jinsoo Kim, and Lohendran Baskaran (2026).

After reading and studying this paper, I decided to implement its proposed methodology step by step, with the objective of understanding the architecture, reproducing its main components, and investigating how its different mechanisms contribute to 3D medical image classification.

The paper introduces a multi-view architecture that exploits complementary anatomical information from three orthogonal planes: axial, coronal, and sagittal.

The proposed approach combines pretrained medical image representations, plane-specific processing, intra-plane Transformers, and axial-centric cross-plane attention.

Beyond implementing the original methodology, this project also explores an additional research direction: **investigating whether the representations learned through this architecture can support few-shot classification when only a limited number of labeled examples are available.**

The project therefore has two complementary objectives:

- **Original research implementation:** Understand, implement, and experimentally explore the architecture proposed by Park et al.
- **Additional investigation:** Evaluate the few-shot learning capabilities of the extracted and learned representations.

## 2. Scientific Background and Motivation

### 2.1 Understanding Multi-View 3D Medical Image Representation

Three-dimensional medical volumes contain rich anatomical and spatial information.

They can be examined through three orthogonal anatomical planes:

- **Axial:** Horizontal cross-sectional views.
- **Coronal:** Frontal views.
- **Sagittal:** Side views.

Each plane provides a different perspective of the same anatomical structure.

A central challenge is how to exploit the complementary information available across these views to learn meaningful representations of a 3D medical volume.

Processing each plane independently may limit the ability to capture relationships between different anatomical perspectives.

The reference paper addresses this challenge through an axial-centric cross-plane attention mechanism.

### 2.2 The Main Idea of the Reference Paper

The authors propose an architecture in which the axial representation serves as the central representation, while information from the coronal and sagittal planes is incorporated through cross-plane attention.

The architecture aims to model interactions between anatomical views rather than simply processing or combining them independently.

Its main components include:

| Component | Role |
|---|---|
| MedDINOv3 | Pretrained medical feature extraction |
| RICA blocks | Plane-specific representation processing |
| Intra-plane Transformers | Modeling relationships between slices within each plane |
| Axial-centric cross-plane attention | Integrating complementary information across anatomical planes |
| Classification head | Predicting the target class |

Understanding the purpose and interaction of these components is a central part of this implementation.

## 3. Reference Research Paper

### Axial-Centric Cross-Plane Attention for 3D Medical Image Classification

**Authors:** Doyoung Park, Jinsoo Kim, Lohendran Baskaran.

**Year:** 2026.

**Original publication:**

- [Read the paper on arXiv](https://arxiv.org/abs/2602.21636)
- [Download the PDF](https://arxiv.org/pdf/2602.21636)

This paper serves as the primary scientific reference for the project.

The implementation will follow the methodology described by the authors as closely as possible, while documenting any differences caused by dataset characteristics, available resources, or implementation constraints.

The goal is to understand the proposed architecture through implementation and experiments, rather than treating it as a black-box model.

## 4. Research Objectives

### Objective 1 — Reproduce and Explore the Original Architecture

- Study the proposed methodology and architectural design.
- Reproduce the pretrained MedDINOv3 feature extraction process.
- Implement the plane-specific representation learning components.
- Understand and implement intra-plane Transformer processing.
- Implement axial-centric cross-plane attention.
- Integrate the classification pipeline.
- Evaluate the implementation and compare results with the reference paper where experimental settings are comparable.

### Objective 2 — Investigate Few-Shot Learning on Learned Representations

An additional direction of this project is to explore whether representations learned through the proposed multi-view architecture can remain discriminative when only a small number of labeled examples are available.

The investigation will focus on:

- Evaluating MedDINOv3 representations under few-shot settings.
- Investigating few-shot classification using multi-view representations.
- Evaluating representations obtained after cross-plane attention.
- Comparing different representation strategies under limited supervision.
- Studying the effect of different numbers of labeled examples per class.

Possible experimental settings include 1-shot, 5-shot, and 10-shot classification.

**Important:** Few-shot learning is an additional investigation proposed in this project. It is not presented as a component or contribution of the original reference paper.

## 5. Dataset

### OrganMNIST3D — MedMNIST

The initial experiments use OrganMNIST3D, part of the MedMNIST benchmark.

The dataset consists of 3D abdominal CT-derived organ volumes representing 11 anatomical categories.

| Property | Description |
|---|---|
| Dataset | OrganMNIST3D |
| Source | MedMNIST v3 |
| Number of classes | 11 |
| Training samples | 971 |
| Validation samples | 161 |
| Test samples | 610 |
| Volume dimensions | 64 × 64 × 64 |
| Data type | 3D grayscale volumes |

**Organ categories:**

Liver, kidney-right, kidney-left, femur-right, femur-left, bladder, heart, lung-right, lung-left, spleen, and pancreas.

Resources:

- [MedMNIST Official Website](https://medmnist.com/)
- [MedMNIST GitHub Repository](https://github.com/MedMNIST/MedMNIST)

The initial implementation focuses on OrganMNIST3D. Additional datasets may be considered in subsequent experiments.

## 6. Methodology and Implementation Pipeline

The implementation is organized into progressive stages.

### Part 1 — Frozen MedDINOv3 Feature Extraction

**Status: Completed**

The first stage establishes the feature extraction pipeline using a pretrained MedDINOv3 ViT-B/16 CT-3M model.

The foundation model is kept frozen and used to extract slice-level representations from the three anatomical views.

**Main steps:**

1. Load the official OrganMNIST3D training, validation, and test splits.
2. Explore and verify the dataset structure.
3. Apply preprocessing and dataset-specific normalization.
4. Extract axial, coronal, and sagittal slice sequences.
5. Resize individual slices to 224 × 224.
6. Prepare three-channel inputs from grayscale slices.
7. Load and validate the pretrained MedDINOv3 checkpoint.
8. Extract normalized CLS token representations.
9. Validate and save the extracted feature tensors.

**Configuration:**

| Parameter | Value |
|---|---|
| Foundation model | MedDINOv3 ViT-B/16 CT-3M |
| Model status | Frozen |
| Embedding dimension | 768 |
| Slices per view | 64 |
| Slice resolution | 224 × 224 |
| Number of views | 3 |

**Extracted feature dimensions:**

| Split | Feature shape per view |
|---|---|
| Training | (971, 64, 768) |
| Validation | (161, 64, 768) |
| Testing | (610, 64, 768) |

Each volume is represented by 64 slice-level feature vectors per anatomical view, with 768 dimensions per vector.

### Part 2 — Plane-Specific Representation Learning

**Status: Planned**

This stage focuses on the implementation of the plane-specific processing components described in the reference paper.

It will investigate:

- RICA blocks.
- Plane-specific positional and contextual information.
- Intra-plane Transformer encoders.
- Slice-level representation processing.

The objective is to understand how the architecture learns meaningful contextual representations within each anatomical plane.

### Part 3 — Axial-Centric Cross-Plane Attention

**Status: Planned**

This stage focuses on implementing the central cross-plane attention mechanism proposed by the authors.

The investigation will cover:

- Relationships between axial, coronal, and sagittal representations.
- Axial-centric attention.
- Cross-plane information exchange.
- Integration of complementary anatomical information.

The objective is to understand how the model combines information across different anatomical views.

### Part 4 — Classification and Evaluation

**Status: Planned**

This stage integrates the learned representations into the classification pipeline.

Experiments will investigate:

- Classification head implementation.
- Model training and validation.
- Classification performance.
- Comparison with relevant baselines.
- Evaluation under comparable experimental settings.

Potential metrics include accuracy, precision, recall, F1-score, and AUC.

### Part 5 — Few-Shot Learning Exploration

**Status: Planned**

After establishing the original architecture and obtaining its learned representations, an additional investigation will explore their effectiveness in few-shot classification.

The objective is to determine whether the representations learned through multi-view processing and cross-plane attention can support classification with limited labeled data.

Potential experiments include:

- Few-shot classification using frozen MedDINOv3 features.
- Few-shot classification using multi-view representations.
- Few-shot classification using cross-plane attention representations.
- Comparison of 1-shot, 5-shot, and 10-shot settings.
- Comparison of different few-shot classification strategies.

This stage will help investigate the discriminative quality and transferability of the learned representations.

### Part 6 — Experimental Analysis and Ablation Studies

**Status: Planned**

Further experiments will investigate the contribution of different architectural components.

Potential analyses include:

- Individual anatomical view contribution.
- Multi-view versus single-view representations.
- Contribution of cross-plane attention.
- Effect of different feature fusion strategies.
- Architectural ablation studies.
- Comparison of standard classification and few-shot settings.

The aim is to understand how the architecture behaves and which components contribute most to its performance.

## 7. Repository Structure

```text
3D-Medical-Image-Representation-Learning/
│
├── README.md
│
├── notebooks/
│   ├── 01_MedDINOv3_Feature_Extraction.ipynb
│   ├── 02_Plane_Specific_Representation_Learning.ipynb
│   ├── 03_Cross_Plane_Attention.ipynb
│   ├── 04_Classification_and_Evaluation.ipynb
│   └── 05_Few_Shot_Learning_Exploration.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── attention/
│   ├── training/
│   └── evaluation/
│
├── results/
│   ├── figures/
│   └── experiments/
│
└── requirements.txt
```

The repository will be updated progressively as each stage is implemented.

## 8. Technologies and Tools

- Python
- PyTorch
- MedDINOv3
- MedMNIST
- NumPy
- Scikit-learn
- Kaggle
- Git and GitHub

## 9. Research Progress

| Stage | Description | Status |
|---|---|---|
| 01 | Reading and analyzing the reference paper | Completed |
| 02 | Dataset exploration and verification | Completed |
| 03 | MedDINOv3 checkpoint validation | Completed |
| 04 | Three-view feature extraction | Completed |
| 05 | Feature validation and storage | Completed |
| 06 | RICA and plane-specific processing | Planned |
| 07 | Intra-plane Transformer implementation | Planned |
| 08 | Axial-centric cross-plane attention | Planned |
| 09 | Classification and evaluation | Planned |
| 10 | Few-shot learning exploration | Planned |
| 11 | Experimental analysis and ablations | Planned |

## 10. Reproducibility and Implementation Notes

The project aims to maintain a transparent and reproducible experimental workflow.

The notebooks will document:

- Dataset configuration and preprocessing.
- Model checkpoints and configurations.
- Feature extraction procedures.
- Architectural implementation details.
- Training and evaluation settings.
- Experimental results and observations.

The implementation will distinguish between the original methodology described by the authors and any adaptations required by the available dataset or computational environment.

Large datasets, pretrained checkpoints, and generated feature tensors are not included directly in this repository.

## 11. References

### Primary Research Paper

Park, D., Kim, J., & Baskaran, L. (2026). *Axial-Centric Cross-Plane Attention for 3D Medical Image Classification.*

- [arXiv paper](https://arxiv.org/abs/2602.21636)
- [PDF](https://arxiv.org/pdf/2602.21636)

### Related Resources

- [MedDINOv3 — Official GitHub Repository](https://github.com/ricklisz/MedDINOv3)
- [MedMNIST — Official Website](https://medmnist.com/)
- [MedMNIST — Official GitHub Repository](https://github.com/MedMNIST/MedMNIST)

---

<p align="center">
  <strong>Read · Understand · Implement · Reproduce · Explore</strong>
  <br>
  3D Medical Image Classification
</p>
