# Pretrained CrP Prediction Models

Pretrained models for performing next-day CRP predictions after the start of antibiotic therapy.

## Using the Docker Image (with Mounted Python Files)

This project provides a prebuilt Docker image designed to be used with mounted Python scripts and configuration files. The image does not include these files directly to allow for potential adaptions to the data extraction or modelling steps. Config files and Python files must be provided via volume mounts when running the container.

### 1. Pull the Docker Image

```bash
docker pull gernotpuc/crp-pipeline:script_v4
```

### 2. Run the Docker Container with Mounted Files

Replace the local paths below with your actual file locations:

```bash
docker run \
  -v /YOUR_PATH/crp_cohort_extraction.py:/app/crp_cohort_extraction.py \
  -v /YOUR_PATH/config_crp.yaml:/app/config_crp.yaml \
  -v /YOUR_PATH/env_py.env:/app/env_py.env \
  -v /YOUR_PATH/results:/app/results \
  gernotpuc/crp-pipeline:script_v4 \
  python crp_cohort_extraction.py --results_dir /app/results
```

Note: This setup requires that the Python script, configuration file, environment variables file, and output directory be provided at runtime using Docker's volume mounting (`-v` option). These files are not included in the Docker image.

## Using the Forecasting Notebook

The notebook `crp_forecasts` enables next-day CRP prediction based on patient-specific laboratory and antibiotic administration data.

### Outputs:

- Performance metrics
- DataFrames with predicted and actual CRP values

## Introduction

Internationally, and also in Germany, the prevalence of multi-resistant pathogens, which do not respond to antibiotics or only respond to them to a limited extent, is increasing. The main cause is the incorrect prescription and application of antibiotics, in particular the unnecessary and excessively long treatment with broad-spectrum antibiotics.

In addition to vital parameters, inflammation levels, especially C-reactive protein (CrP), are the most important clinical parameters for sepsis. However, the response of CRP levels to antibiotic treatment is delayed due to a half-life of 19 hours. Thus, despite overall clinical improvement, an increase in CrP is often still detected, which can lead to an (unnecessary) intensification of antibiotic therapy. The reliable prediction of CrP can therefore contribute to a more restrictive antibiotic therapy.

Using the FHIR data set with a cohort of 2,823 patients at the University Hospital Essen, various monocentric, global time series models were trained, which predict patient-specific CrP values. Models were intentionally kept simple in order to keep the data requirements as low as possible. Thus, no covariates were used here and predictions are made only based on the historic CrP values and time stamps of administered antibiotics.

## Model Performance

Performance was evaluated on an internal dataset of 2,823 time series using 5-fold cross-validation with 20 repeats.

| Model                                                             | MAE  | RMSE | MSE   | MAPE  |
|-------------------------------------------------------------------|------|------|-------|-------|
| Gradient Boosting (LightGBM)                                      | 3.35 | 4.73 | 22.40 | 35.59 |
| Neural Network (NBEATS)                                           | 3.30 | 4.80 | 23.06 | 30.58 |
| Weighted Ensemble of NNs (DeepAR, TiDE, PatchTST, DLinear, TFTM)  | 3.67 | 5.13 | 26.39 | 36.64 |
| Zero-shot Large Language Model (Chronos)                          | 3.96 | 5.65 | 32.01 | 39.13 |
| Baseline model: Average Forecast                                  | 7.70 | 10.72| 98.53 | 60.69 |
| Baseline model: Naive Forecast                                    | 4.82 | 7.18 | 26.88 | 42.47 |

## Cohort Definition

The patient cohort is defined according to the following criteria:

- Patient is an inpatient
- Patient is at least 18 years old
- Cancer diagnosis with ICD Code C00–96, registered after 01-01-2020
- Availability of CRP values during hospital stays.
- IV-administered antibiotics with an iv. antibiotic ATC code.

## Requirements

- Docker must be installed on your system: [Install Docker](https://docs.docker.com/get-docker/)
- The required Python environment is already included in the Docker image, so no additional installation steps are necessary.


## Project Structure

```
├── crp_cohort_extraction.py      # Python script (mounted)
├── config_crp.yaml               # Config file (mounted)
├── env_py.env                    # Environment variables (mounted)
├── results/                      # Output directory (mounted)
├── crp_forecasts.ipynb           # Jupyter notebook for prediction
├── requirements.txt              # Dependency list
└── README.md                     # Project documentation
```

## Contact

For questions, feedback, or issues, please use GitHub Issues or contact the repository maintainer directly.
