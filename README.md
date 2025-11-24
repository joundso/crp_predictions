# Pretrained CrP Prediction Models

Pretrained models for performing next-day CRP predictions after the start of antibiotic therapy.

## Using the Docker Image (with Mounted Python Files)

This project provides a prebuilt Docker image designed to be used with mounted Python scripts and configuration files. The image does not include these files directly to allow for potential adaptions to the data extraction or modelling steps. Config files and Python files must be provided via volume mounts when running the container.

### 1. Pull the Docker Image

For Linux with arm64
```bash
docker pull gernotpuc/crp-pipeline:script_v5
```

For Linux with amd64
```bash
docker pull gernotpuc/crp-pipeline:script_v6
```

### 2. Run the Docker Container with Mounted Files

Replace the local paths below with your actual file locations:

```bash
docker run \
  -v /YOUR_PATH/crp_cohort_extraction.py:/app/crp_cohort_extraction.py \
  -v /YOUR_PATH/config_crp.yaml:/app/config_crp.yaml \
  -v /YOUR_PATH/env_py.env:/app/env_py.env \
  -v /YOUR_PATH/results:/app/results \
  gernotpuc/crp-pipeline:script_v6 \
  python crp_cohort_extraction.py --results_dir /app/results
```

Note: This setup requires that the Python script, configuration file, environment variables file, and output directory be provided at runtime using Docker's volume mounting (`-v` option). These files are not included in the Docker image.

## Using the Cohort extraction and Forecasting Python File

The Python file `crp_cohort_extraction` enables the cohort extraction and next-day CRP prediction based on patient-specific laboratory and antibiotic administration data. Adjust the config_crp.yaml and env_py.env file to fit your FHIR database.

### Expected Data Inputs:

The queried clinical data is expected to be returned in JSON using the FHIR (Fast Healthcare Interoperability Resources) standard.

### Expected Model Outputs:

- CSV-File of Performance metrics
- CSV-File with predicted and actual CRP values

## Introduction

Internationally, and also in Germany, the prevalence of multi-resistant pathogens, which do not respond to antibiotics or only respond to them to a limited extent, is increasing. The main cause is the incorrect prescription and application of antibiotics, in particular the unnecessary and excessively long treatment with broad-spectrum antibiotics.

In addition to vital parameters, inflammation levels, especially C-reactive protein (CrP), are the most important clinical parameters for sepsis. However, the response of CRP levels to antibiotic treatment is delayed due to a half-life of 19 hours. Thus, despite overall clinical improvement, an increase in CrP is often still detected, which can lead to an (unnecessary) intensification of antibiotic therapy. The reliable prediction of CrP can therefore contribute to a more restrictive antibiotic therapy.

Using the FHIR data set with a cohort of 2,823 patients at the University Hospital Essen, various monocentric, global time series models were trained, which predict patient-specific CrP values. Models were intentionally kept simple in order to keep the data requirements as low as possible. Thus, no covariates were used here and predictions are made only based on the historic CrP values and time stamps of administered antibiotics.

## Model Performance

Performance was evaluated on an internal dataset of 2,823 time series using 5-fold cross-validation with 20 repeats.

| Model                                                             | MAE (95% CI)  | RMSE (95% CI) | MSE (95% CI)   | SMAPE (95% CI)  |
|-------------------------------------------------------------------|------|------|-------|-------|
| DeepAR (Deep Learning)                                            | 2.54 (2.08-3.04) | 4.20 (3.44-5.02) | 17.81 (11.85-25.19) | 20.54 (17.88-23.40) |
| TiDE (Deep Learning)                                              | 2.50 (2.05-2.98) | 4.13 (3.34-4.96) | 17.25 (11.16-24.62) | 20.00 (17.56-22.60) |
| PatchTST (Deep Learning)                                          | 2.49 (2.04-3.00) | 4.14 (3.33-4.91) | 17.28 (11.10-23.99) | 20.04 (17.62-22.68) |
| Chronos fine tuned (Foundation model)                             | 2.47 (1.88-3.13) | 3.98 (3.05-4.97) | 16.10 (9.31-24.67)  | 20.61 (17.00-24.67) |
| Chronos zero shot (Foundation model)                              | 2.65 (2.09-3.28) | 4.01 (3.21-4.88) | 16.32 (10.29-23.80) | 22.02 (18.32-25.83) |
| Statistical model: AutoARIMA                                      | 3.22 (2.64-3.80) | 5.07 (4.22-5.96) | 25.90 (17.80-35.59) | 26.43 (22.58-30.31) |
| Baseline model: Average Forecast                                  | 3.79 (3.11-4.45) | 5.87 (4.81-6.93) | 34.78 (23.12-47.98) | 29.72 (26.59-32.91) |


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

## Authentication config in env_py.env

There are two authentication methods supported by the script:
- basic_auth
- token (e.g. with a separate auth server / API gateway)

Depending on the method, different URL variables from env_py.env are used.
The authentication method is defined in the YAML file.

When Basic Authentication is used:
```
authentication_method:
  - 'basic_auth'
```

In this case, the following variables are used:
- FHIR_USER / FHIR_PASSWORD

Username and password for HTTP Basic Auth (read from environment variables because auth_method="env").
- FHIR_SERVER_URL

Base URL of the FHIR server, e.g.:
https://my-server.com/fhir or
https://my-server.com/app/FHIR/r4/

When token Authentication is used:
```
authentication_method:
  - 'token'
```

In this case, the variables mean:
- SEARCH_URL

FHIR base URL, e.g.:
https://my-server.com/app/FHIR/r4/
- BASIC_AUTH

URL of the auth endpoint used to obtain the token, e.g.:
https://my-server.com:8443/app/Auth/v1/basicAuth
- REFRESH_AUTH

URL of the token refresh endpoint (if a separate endpoint exists), e.g.:
https://my-server.com:8443/app/Auth/v1/refreshToken

## Project Structure

```
├── crp_cohort_extraction.py      # Python script (mounted)
├── config_crp.yaml               # Config file (mounted)
├── env_py.env                    # Environment variables (mounted)
├── results/                      # Output directory (mounted)
├── requirements.txt              # Dependency list
└── README.md                     # Project documentation
```

## Contact

For questions, feedback, or issues, please use GitHub Issues or contact the repository maintainer directly.
