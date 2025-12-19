#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CRP Forecasting Pipeline using FHIR-Pyrate + AutoGluon.
Cleansed and modularized version for improved readability and maintainability.
"""

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from fhir_pyrate import Ahoy, Pirate
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def init_authentication(config: dict):
    """Initialize FHIR authentication and return a Pirate search object."""
    load_dotenv(dotenv_path="/app/env_py.env")
    auth_method = config["authentication_method"][0]

    if auth_method == "basic_auth":
        fhir_user = os.environ["FHIR_USER"]
        fhir_url = os.environ["FHIR_SERVER_URL"]

        auth = Ahoy(
            auth_url=fhir_url,
            auth_type="BasicAuth",
            username=fhir_user,
            auth_method="env",
        )

        return Pirate(fhir_url, auth=auth, print_request_url=False)

    else:
        basic_auth = os.environ["BASIC_AUTH"]
        refresh_auth = os.environ["REFRESH_AUTH"]
        search_url = os.environ["SEARCH_URL"]

        auth = Ahoy(
            auth_type="token",
            auth_method="env",
            auth_url=basic_auth,
            refresh_url=refresh_auth,
        )

        return Pirate(
            base_url=search_url, auth=auth, print_request_url=False
        )


def fetch_fhir_dataframe(search: Pirate, df: pd.DataFrame, resource: str,
                         params: dict, constraints: dict) -> pd.DataFrame:
    """Fetch FHIR bundles and convert them to a DataFrame."""
    bundles = search.trade_rows_for_bundles(
        df, resource_type=resource, request_params=params, df_constraints=constraints
    )
    return search.bundles_to_dataframe(bundles=bundles)


# -------------------------------------------------------------------------
# Data Processing Functions
# -------------------------------------------------------------------------


def compute_patient_age(encounters_df, patients_df, birthdate_field):
    """Merge encounters with patient birthdates and compute ages."""
    patients_df[birthdate_field] = pd.to_datetime(patients_df[birthdate_field], errors="coerce")

    encounters_df["period_start"] = pd.to_datetime(encounters_df["period_start"], utc=True)
    encounters_df["period_start"] = encounters_df["period_start"].dt.tz_localize(None)

    encounters_df["subject_identifier"] = encounters_df["subject_reference"].str.replace("Patient/", "")

    merged = pd.merge(
        encounters_df,
        patients_df[["id", birthdate_field]],
        left_on="subject_identifier",
        right_on="id",
        how="left",
    )

    merged["age"] = (merged["period_start"] - merged[birthdate_field]).dt.days // 365
    return merged


def resample_time_series(df, rate, min_length, impute, max_train_length, value_col):
    """Resample CRP values per encounter and filter short time series."""

    df["effectiveDateTime"] = pd.to_datetime(df["effectiveDateTime"], utc=True)
    df = df.dropna(subset=["effectiveDateTime"])

    agg = {value_col: "max"}

    def process_group(group):
        group = group.set_index("effectiveDateTime")
        resampled = group.resample(rate).agg(agg)
        resampled["encounter_reference"] = group["encounter_reference"].iloc[0]
        resampled.reset_index(inplace=True)

        # Imputation
        if impute == "interpolate":
            resampled[value_col] = resampled[value_col].interpolate(method="linear")
        elif impute == "forward_fill":
            resampled[value_col] = resampled[value_col].ffill().bfill()

        return resampled

    df = df.groupby("encounter_reference").apply(process_group).reset_index(drop=True)
    df = df.sort_values(["encounter_reference", "effectiveDateTime"])

    df = df.groupby("encounter_reference").tail(max_train_length)

    valid_ids = df["encounter_reference"].value_counts()
    valid_ids = valid_ids[valid_ids >= min_length].index

    return df[df["encounter_reference"].isin(valid_ids)]


def compute_bootstrap_metrics(actual, predicted, n_bootstrap=1000, ci_percentile=95):
    """Compute bootstrap confidence intervals for regression metrics."""
    mae_list, mse_list, rmse_list, mape_list, smape_list = [], [], [], [], []

    n = len(actual)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        a, p = actual[idx], predicted[idx]

        mae = np.mean(np.abs(a - p))
        mse = np.mean((a - p) ** 2)
        rmse = np.sqrt(mse)
        mape = (np.abs(a - p) / np.abs(a)).mean() * 100
        smape = (np.abs(a - p) / (np.abs(a) + np.abs(p))).mean() * 100

        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mape_list.append(mape)
        smape_list.append(smape)

    def ci(values):
        return (
            np.mean(values),
            np.percentile(values, (100 - ci_percentile) / 2),
            np.percentile(values, 100 - (100 - ci_percentile) / 2),
        )

    return {
        "mae": ci(mae_list),
        "mse": ci(mse_list),
        "rmse": ci(rmse_list),
        "mape": ci(mape_list),
        "smape": ci(smape_list),
    }


# -------------------------------------------------------------------------
# AutoGluon Forecasting Function
# -------------------------------------------------------------------------


def autogluon_ci_prediction(model_name, output_path, df, predictor,
                            prediction_length, target_col):
    """Generate predictions with confidence intervals and return metrics."""

    df["effectiveDateTime"] = df["effectiveDateTime"].dt.tz_localize(None)

    ts = TimeSeriesDataFrame.from_data_frame(
        df, id_column="encounter_reference", timestamp_column="effectiveDateTime"
    )

    # Split last N points for test set
    train, test = [], []
    for _, g in df.groupby("encounter_reference"):
        test_rows = g.nlargest(prediction_length, "effectiveDateTime")
        train_rows = g.drop(test_rows.index)
        train.append(train_rows)
        test.append(test_rows)

    train = TimeSeriesDataFrame.from_data_frame(pd.concat(train), "encounter_reference", "effectiveDateTime")
    test = TimeSeriesDataFrame.from_data_frame(pd.concat(test), "encounter_reference", "effectiveDateTime")

    preds = predictor.predict(train, model=model_name)[["mean", "0.1", "0.9"]]

    merged = test.join(preds, how="inner")
    actual = merged[target_col].values.reshape(-1)
    predicted = merged["mean"].values.reshape(-1)

    metrics = compute_bootstrap_metrics(actual, predicted)

    pd.DataFrame([
        {"Metric": m.upper(), "Estimate": v[0], "CI_Lower": v[1], "CI_Upper": v[2]}
        for m, v in metrics.items()
    ]).to_csv(output_path, index=False)

    return metrics


# -------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    config = load_config("config_crp.yaml")
    search = init_authentication(config)

    fhir_count = str(config.get("fhir_count", 100))

    # ---------------------------------------------------------------------
    # FHIR Queries
    # ---------------------------------------------------------------------

    conditions_df = fetch_fhir_dataframe(
        search,
        pd.DataFrame(config["icd_codes"], columns=["icd-10"]),
        resource="Condition",
        params={"_count": fhir_count, "_sort": "_id", "recorded-date": "ge2022-01"},
        constraints={"code": "icd-10"},
    )

    conditions_unique = conditions_df.dropna(subset=["encounter_reference"]).drop_duplicates("encounter_reference")

    encounters_df = fetch_fhir_dataframe(
        search,
        conditions_unique,
        resource="Encounter",
        params={"_count": fhir_count, "_sort": "_id"},
        constraints={"_id": "encounter_reference"},
    )

    # Filter encounter class & diagnosis type if present
    stay_field = config["encounter_stay_type_field"][0]
    diag_field = config["encounter_diagnosis_type_field"][0]

    if stay_field in encounters_df:
        encounters_df = encounters_df[encounters_df["class_code"].isin(config["encounter_stay_type_content"])]

    if diag_field in encounters_df:
        encounters_df = encounters_df[encounters_df[diag_field].isin(config["encounter_diagnosis_type_content"])]

    # ---------------------------------------------------------------------
    # Medication Filtering
    # ---------------------------------------------------------------------

    meds_df = fetch_fhir_dataframe(
        search,
        pd.DataFrame(config["medication_codes"], columns=["atc_codes"]),
        "Medication",
        {"_count": fhir_count, "_sort": "_id"},
        {"code": "atc_codes"},
    )
    meds_df["medication_reference"] = "Medication/" + meds_df["id"].astype(str)

    admin_df = fetch_fhir_dataframe(
        search,
        encounters_df,
        "MedicationAdministration",
        {"_count": fhir_count, "_sort": "_id"},
        {"context": "id"},
    )

    admin_df = admin_df[
        admin_df["medicationReference_reference"].isin(meds_df["medication_reference"])
    ]

    # ---------------------------------------------------------------------
    # Patient Filtering
    # ---------------------------------------------------------------------

    patients_df = fetch_fhir_dataframe(
        search,
        admin_df.drop_duplicates("subject_reference"),
        "Patient",
        {"_count": fhir_count, "_sort": "_id"},
        {"_id": "subject_reference"},
    )

    encounters_with_age = compute_patient_age(
        encounters_df, patients_df, config["patients_birthdate_field"][0]
    )
    encounters_with_age = encounters_with_age[encounters_with_age["age"] >= 18]

    # ---------------------------------------------------------------------
    # CRP Extraction
    # ---------------------------------------------------------------------

    crp_df = fetch_fhir_dataframe(
        search,
        encounters_with_age,
        "Observation",
        {"_count": fhir_count, "_sort": "date", "code": config["crp_laboratory_code"]},
        {"encounter": "id"},
    )

    if config["crp_unit"][0] == "mg/l":
        crp_df["valueQuantity_value"] /= 10

    # ---------------------------------------------------------------------
    # Time-Series Preparation
    # ---------------------------------------------------------------------

    predictor = TimeSeriesPredictor.load("crp_ensemble", require_version_match=False)

    resampled = resample_time_series(
        crp_df,
        rate="1d",
        min_length=4,
        impute="forward_fill",
        max_train_length=14,
        value_col="valueQuantity_value",
    )

    os.makedirs(args.results_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Forecasting
    # ---------------------------------------------------------------------

    for model_name in predictor.model_names():
        output_path = os.path.join(args.results_dir, f"{model_name}_metrics.csv")
        print(f"Running model: {model_name}")
        metrics = autogluon_ci_prediction(
            model_name,
            output_path,
            resampled,
            predictor,
            prediction_length=1,
            target_col="valueQuantity_value",
        )
        print(f"Saved metrics for {model_name} → {output_path}")

    print("All done. Thank you. You are awesome.")


# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
