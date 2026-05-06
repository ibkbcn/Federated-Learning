# Federated Learning vs Centralized Training: Epileptic Seizure Detection

A comparative study of federated and centralized machine learning for binary classification of epileptic seizures using EEG brain frequency data, exploring privacy-preserving training in the context of EU data regulation.

## Why This Matters

Healthcare data is among the most sensitive and fragmented. Hospitals can't easily share patient records, yet ML models need large datasets to perform well. **Federated Learning** offers a solution: train models collaboratively without centralizing the data.

This project investigates the **real cost** of that privacy gain, how much performance do we lose, and where exactly does federated training struggle?

## The Experiment

**Dataset:** [Epileptic Seizure Recognition](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition) — 11,500 EEG signal windows (178 features each), binary classification: seizure vs. non-seizure.

**Models tested:**
- **MLP** (Multi-Layer Perceptron) with Binary Cross-Entropy
- **Linear SVM** (SGD Classifier) with Hinge Loss

**Training setups compared across 4 simulations:**

| Simulation | Model | Data Distribution | Loss Function |
|:---:|:---:|:---:|:---:|
| 1 | MLP | IID | Binary Cross-Entropy |
| 2 | MLP | Non-IID | Binary Cross-Entropy |
| 3 | SVM | IID | Hinge |
| 4 | SVM | Non-IID | Hinge |

- **Centralized**: standard single-node training (1,000 epochs)
- **Federated**: 10 nodes, FedAvg aggregation, 200 rounds × 5 local epochs
- **IID**: data split randomly (balanced across nodes)
- **Non-IID**: data split via Dirichlet distribution (α=0.5), creating realistic heterogeneity
- Each simulation averaged over **5 independent runs**

## Key Findings

### MLP performed well overall, but federated training showed consistent gaps

| Setup | Accuracy | Precision | Recall | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|
| Centralized | 0.948 | 0.916 | 0.812 | 0.948 |
| Federated IID | 0.941 | 0.928 | 0.764 | 0.939 |
| Federated Non-IID | 0.900 | 0.970 | 0.518 | 0.819 |

### Federated models develop a majority-class bias

The most critical finding: federated training systematically favors the majority class (non-seizure), **reducing recall for actual seizure cases**. In a medical context, this means more missed diagnoses.

The Non-IID federated MLP dropped recall to **0.518** — nearly half the seizures were missed, despite maintaining high accuracy (0.90) due to the class imbalance.

![Confusion Matrix Comparison](assets/confusion_matrix_comparison.png)

### The generalization gap is inherent to federation, not just Non-IID

Even in the IID scenario, the gap between training and validation performance was larger for federated models than centralized ones. This suggests the FedAvg aggregation process itself introduces generalization challenges beyond data heterogeneity.

### SVM struggled regardless of training approach

Both centralized and federated SVM achieved ROC-AUC ~0.52 (near random), indicating the linear model couldn't capture the underlying patterns in EEG data. Interestingly, the federated Non-IID SVM slightly outperformed its centralized counterpart.

## Tech Stack

- **Python** — core language
- **PyTorch** — model implementation (MLP and SVM with gradient descent)
- **Flower (flwr)** — federated learning simulation framework
- **scikit-learn** — preprocessing, metrics, validation
- **Pandas / NumPy** — data manipulation
- **Matplotlib** — visualization

## Project Structure

├── CÓDIGO Ivan Betriu.ipynb    # Full experiment notebook
├── TFM Ivan Betriu.pdf         # Complete thesis document (Spanish)
├── assets/                     # Figures and visualizations
└── README.md

## Regulatory Context

This work is framed within the **EU Data Strategy (2021-2027)**, analyzing how federated learning aligns with the Data Governance Act, Data Act, and GDPR requirements for privacy-preserving data analysis across institutions.

## Limitations & Future Work

- Models used are intentionally simple, the goal was comparing training paradigms, not maximizing performance
- Only FedAvg was tested; advanced aggregation methods (FedProx, SCAFFOLD, FedMA) could mitigate the issues found
- A single dataset was used; cross-domain validation would strengthen conclusions
- Class imbalance mitigation techniques in federated settings remain unexplored

## Author

**Ivan Betriu Kahlenberg**
Master's Thesis — M.Sc. Data Science, La Salle (Universitat Ramon Llull), 2025

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/ivan-betriu-kahlenberg)
