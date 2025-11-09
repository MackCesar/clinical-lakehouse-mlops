# Clinical + Economic AI Lakehouse

This project sets up a modern data and AI platform for healthcare analytics using open-source tools.
It combines Spark, MLflow, and Docker for local development, and includes Terraform templates for cloud deployment.

## Overview

The goal is to build an end-to-end system for:

Clinical and cost prediction using structured and unstructured healthcare data

Readmission risk models (XGBoost / LightGBM)

Clinical note summarization (BioClinicalBERT)

Infrastructure that supports MLOps, governance, and Responsible AI

### Tech Stack
| **Layer**     | **Tools**                          |
|:--------------|:-----------------------------------|
| Data          | Spark • Delta Lake • MinIO          |
| Modeling      | XGBoost • BioClinicalBERT           |
| MLOps         | MLflow • Docker • Terraform         |
| Monitoring    | Prometheus • Grafana                |
| Governance    | FHIR validation                     |


### Quick Start

1. Generate sample data

```bash
brew install openjdk@17
brew install synthea

synthea -p 1000 --exporter.csv.export true
mv output/csv/* data/mimic_synthetic/
```

2. Run locally
```bash
make up
# Spark UI: http://localhost:4040
# MLflow:   http://localhost:5000
```

3 .Shut down
```bash
make down
```

### Project Structure
```
clinical-lakehouse-mlops/
├── data/
├── notebooks/
├── pipelines/
├── models/
├── monitoring/
├── serving/
├── infra/
│   ├── docker/
│   └── terraform/
├── Makefile
└── README.md
```
### License

MIT License © 2025