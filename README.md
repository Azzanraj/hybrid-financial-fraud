# Hybrid Deep Learning Framework for Financial Fraud Detection

A major project for detecting fraudulent financial transactions using a hybrid multi-view learning pipeline that combines **Autoencoder (AE)**, **Graph Attention Network (GAT)**, and **LightGBM**. The system is designed for **IEEE-CIS style transaction data** and produces a unified fraud risk score through model fusion.

## Overview

Financial fraud detection is difficult because transaction data is highly imbalanced, fraud patterns evolve over time, and many suspicious activities are hidden across relationships between cards, devices, addresses, and transaction histories. This project addresses those issues by combining three complementary views:

- **Tabular learning** with **LightGBM** to capture non-linear patterns in structured transaction features.
- **Relational learning** with **GAT** to model links between related transactions and entities.
- **Anomaly detection** with **Autoencoder** to learn normal transaction behavior and identify deviations using reconstruction error.

The outputs from these models are fused into a **final fraud score** and thresholded to classify transactions as **Fraud** or **Normal**.

## Key Features

- CSV upload-based fraud detection workflow
- Hybrid model fusion using AE, GAT, and LightGBM
- Final risk score generation
- Adjustable fraud threshold
- Fraud / Normal prediction with risk levels
- Downloadable result CSV
- Flask-based web interface
- Supports IEEE-CIS style transaction datasets
- Pretrained model artifacts stored in the repository

## Project Motivation

Traditional fraud detectors often rely on a single learning perspective. That is usually not enough for modern fraud scenarios, where the same fraud pattern can appear as:

- unusual transaction values,
- repeated activity across shared entities,
- or subtle anomalies hidden in high-dimensional features.

This project was built to improve detection robustness by combining behavioral, relational, and anomaly-based signals in one framework. The project report states that the proposed framework achieved strong performance, including **98.94% ROC-AUC**, **90.67% PR-AUC**, **77.11% precision**, **88.00% recall**, and **82.20% F1-score**.

## System Architecture

```text
Input Transaction CSV
        │
        ▼
Data Preprocessing
        │
 ┌───────────────┬────────────────┬────────────────┐
 ▼               ▼                ▼
Autoencoder      GAT            LightGBM
(AE Score)    (Graph Score)    (LGB Score)
 └───────────────┴────────────────┴────────────────┘
                     │
                     ▼
          Hybrid Risk Score Fusion
                     │
                     ▼
          Fraud / Normal Prediction
                     │
                     ▼
          Flask Web Dashboard
```

The report describes this as a multi-view framework that combines tabular, relational, and anomaly-based learning, and then fuses the outputs into a unified fraud score. fileciteturn1file2turn1file6

## Workflow

1. Upload a transaction CSV through the Flask dashboard.
2. Validate and preprocess the input data.
3. Generate scores from:
   - LightGBM
   - Autoencoder
   - GAT
4. Normalize and combine model outputs using weighted fusion.
5. Apply the selected threshold to generate the final label.
6. Display results in a table and allow CSV download.

The project report also describes score allocation and threshold optimization as part of the fusion pipeline. 

## Technologies Used

### Backend
- Python
- Flask

### Machine Learning / Deep Learning
- LightGBM
- PyTorch
- Graph Attention Network (GAT)
- Autoencoder
- Scikit-learn
- NumPy
- Pandas

### Frontend
- HTML
- CSS

## Repository Structure

```text
.
├── app.py
├── major_code.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
├── models/
├── static/
├── templates/
├── test_datasets/
├── uploads/
└── outputs/
```

### Main folders
- `models/` — saved model files, scalers, feature pickles, graph data, score arrays, and other trained artifacts
- `templates/` — Flask HTML templates for the home page and results page
- `static/` — CSS styling
- `test_datasets/` — sample CSV files for testing
- `uploads/` — uploaded input files during runtime
- `outputs/` — generated fraud result CSV files

## Dataset

The project is built around **IEEE-CIS style transaction data**. The report states that the system is intended for offline analysis using publicly available datasets such as the IEEE-CIS Fraud Detection dataset. 

The project is best used with CSV files that contain the same feature names used during training.

## Model Components

### 1. Autoencoder (AE)
The autoencoder learns normal transaction behavior and detects anomalies from reconstruction error. It is useful for rare and previously unseen patterns.

### 2. Graph Attention Network (GAT)
GAT captures relationships between transactions and shared entities such as card, address, and device features. This helps detect coordinated fraud patterns that are difficult to identify from tabular data alone. 

### 3. LightGBM
LightGBM handles high-dimensional tabular transaction data and learns complex non-linear interactions efficiently. 
## Functional Scope

The project report defines the system requirements as follows:

- ingest financial transaction data,
- preprocess and validate records,
- generate temporal, behavioral, and relational features,
- build a transaction graph,
- train the autoencoder,
- train the hybrid model,
- and evaluate using Precision, Recall, F1-score, and ROC-AUC. 

## Evaluation

The report evaluates the system using standard fraud detection metrics, including:

- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Log Loss
- Balanced Accuracy

The proposed framework is compared against several existing baselines and hybrid methods. The report states that the proposed model outperformed the baselines in the final comparison table. 

## Hardware and Software Requirements

The report lists the following software stack:

- Python 3.10
- PyTorch 2.0.7
- LightGBM 3.3.5
- NumPy 1.24.4
- Pandas 1.5.3
- Matplotlib 3.7.1
- scikit-learn 1.3.0
- Flask 2.3.2

## Installation

```bash
git clone https://github.com/Azzanraj/hybrid-financial-fraud.git
cd hybrid-financial-fraud
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Git LFS Note

A large dataset file is tracked with Git LFS:

- `models/have_up_subset.csv`

This keeps the repository usable on GitHub while preserving the large training subset needed for reproducibility.

## Example Output

The results page shows:

- total rows processed
- predicted fraud count
- fraud rate
- threshold value
- a results table with per-transaction scores
- final label
- risk level
- downloadable CSV output

## Project Goal

The goal of this major project is to provide a practical fraud detection pipeline that combines:

- **tabular learning**
- **graph-based relational learning**
- **anomaly detection**
- **score fusion**

so that suspicious transactions can be identified more reliably than with a single-model system. This objective is consistent with the project abstract and report methodology. 

## Authors

- R. Azzan Prashasth Raj
- Mohammed Fazulul Rakheeb
- Gullapalli Prabhu Kumar
- Kuchipudi Vijay Rahul

## Guide

Dr. P. Surendra Varma, M.Tech, Ph.D  
Sr. Assistant Professor, Department of CSE

## License

This repository is intended for academic and project demonstration use.
