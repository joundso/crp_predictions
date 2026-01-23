#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CRP Forecasting Pipeline using FHIR-Pyrate + AutoGluon.
Now includes proper logging (console + file), with timestamps, log levels,
and exception stack traces.

Usage:
  python crp_pipeline.py --results_dir ./results --log_level INFO

Notes:
- Logs go to:
    1) stdout (good for Docker/K8s log collectors)
    2) <results_dir>/run.log
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from fhir_pyrate import Ahoy, Pirate
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------


def setup_logging(results_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging to both console and file, and return a module logger.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log_level '{log_level}'. Use DEBUG/INFO/WARNING/ERROR/CRITICAL.")

    logger = logging.getLogger("crp_pipeline")
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs if root logger is configured elsewhere

    # Clear existing handlers (important if re-imported / run in notebooks)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(Path(results_dir) / "run.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------


def load_config(config_path: str, logger: logging.Logger) -> Dict:
    """Load YAML configuration."""
    logger.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug("Config keys loaded: %s", list(config.keys()))
    return config


def init_authentication(config: Dict, logger: logging.Logger) -> Pirate:
    """
    Initialize FHIR authentication and return a Pirate search object.
    Expects environment variables to be present in /app/env_py.env.
    """
    env_path = "/app/env_py.env"
    load_dotenv(dotenv_path=env_path)
    logger.info("Loaded environment variables from %s (if present)", env_path)

    auth_method = config["authentication_method"][0]
    logger.info("Authentication method: %s", auth_method)

    if auth_method == "basic_auth":
        try:
            fhir_user = os.environ["FHIR_USER"]
            fhir_url = os.environ["FHIR_SERVER_URL"]
        except KeyError as e:
            logger.error("Missing required environment variable: %s", e)
            raise

        auth = Ahoy(
            auth_url=fhir_url,
            auth_type="BasicAuth",
            username=fhir_user,
            auth_method="env",
        )

        logger.info("Initialized BasicAuth for FHIR server %s", fhir_url)
        return Pirate(fhir_url, auth=auth, print_request_url=False)

    # token-based
    try:
        basic_auth = os.environ["BASIC_AUTH"]
        refresh_auth = os.environ["REFRESH_AUTH"]
        search_url = os.environ["SEARCH_URL"]
    except KeyError as e:
        logger.error("Missing required environment variable: %s", e)
        raise

    auth = Ahoy(
        auth_type="token",
        auth_method="env",
        auth_url=basic_auth,
        refresh_url=refresh_auth,
    )

    logger.info("Initialized token auth for base_url %s", search_url)
    return Pirate(base_url=search_url, auth=auth, print_request_url=False)


def fetch_fhir_dataframe(
    search: Pirate,
    df: pd.DataFrame,
    resource: str,
    params: Dict,
    constraints: Dict,
    logger: logging.Logger,
    label: str = "",
) -> pd.DataFrame:
    """Fetch FHIR bundles and convert them to a DataFrame."""
    label_prefix = f"{label}: " if label else ""

    logger.info("%sFetching FHIR resource=%s params=%s constraints=%s", label_prefix, resource, params, constraints)

    bundles = search.trade_rows_for_bundles(
        df,
        resource_type=resource,
        request_params=params,
        df_constraints=constraints,
    )

    out = search.bundles_to_dataframe(bundles=bundles)
    logger.info("%sFetched %d rows for resource=%s", label_prefix, len(out), resource)
    logger.debug("%sColumns for %s: %s", label_prefix, resource, list(out.columns))
    return out


# -------------------------------------------------------------------------
# Data Processing Functions
# -------------------------------------------------------------------------


def compute_patient_age(encounters_df: pd.DataFrame, patients_df: pd.DataFrame, birthdate_field: str, logger: logging.Logger) -> pd.DataFrame:
    """Attach patient age to encounters without clobbering encounter 'id' column."""
    logger.info("Computing patient ages using birthdate field '%s'", birthdate_field)

    enc = encounters_df.copy()
    pat = patients_df.copy()

    # Patients: parse birthdate
    if birthdate_field not in pat.columns:
        raise KeyError(f"Patient birthdate field '{birthdate_field}' not found in patients_df columns: {list(pat.columns)}")
    pat[birthdate_field] = pd.to_datetime(pat[birthdate_field], errors="coerce")

    # Encounters: parse start time
    if "period_start" not in enc.columns:
        raise KeyError(f"'period_start' not found in encounters_df columns: {list(enc.columns)}")
    enc["period_start"] = pd.to_datetime(enc["period_start"], utc=True, errors="coerce").dt.tz_localize(None)

    # Build patient identifier to join on
    if "subject_reference" not in enc.columns:
        raise KeyError(f"'subject_reference' not found in encounters_df columns: {list(enc.columns)}")
    enc["subject_identifier"] = enc["subject_reference"].astype(str).str.replace("Patient/", "", regex=False)

    # Build a lookup: patient_id -> birthdate
    pat_birth = pat.set_index("id")[birthdate_field]  # uses Patient.id intentionally
    enc["birthDate"] = enc["subject_identifier"].map(pat_birth)

    # Age calculation
    enc["age"] = (enc["period_start"] - enc["birthDate"]).dt.days // 365

    logger.info("Computed ages for %d encounter rows", len(enc))
    return enc



def resample_time_series(
    df: pd.DataFrame,
    rate: str,
    min_length: int,
    impute: str,
    max_train_length: int,
    value_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Resample CRP values per encounter and filter short time series."""
    logger.info(
        "Resampling time series: rate=%s min_length=%d impute=%s max_train_length=%d value_col=%s",
        rate, min_length, impute, max_train_length, value_col
    )

    df = df.copy()
    df["effectiveDateTime"] = pd.to_datetime(df["effectiveDateTime"], utc=True, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["effectiveDateTime"])
    logger.info("Dropped %d rows with missing/invalid effectiveDateTime", before - len(df))

    if value_col not in df.columns:
        raise KeyError(f"Value column '{value_col}' not found in CRP dataframe columns: {list(df.columns)}")

    agg = {value_col: "max"}

    def process_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index("effectiveDateTime")
        resampled = group.resample(rate).agg(agg)
        resampled["encounter_reference"] = group["encounter_reference"].iloc[0]
        resampled.reset_index(inplace=True)

        # Imputation
        if impute == "interpolate":
            resampled[value_col] = resampled[value_col].interpolate(method="linear")
        elif impute == "forward_fill":
            resampled[value_col] = resampled[value_col].ffill().bfill()
        elif impute in ("none", "", None):
            pass
        else:
            # keep behavior explicit
            raise ValueError(f"Unknown impute strategy: {impute!r}")

        return resampled

    df = df.groupby("encounter_reference", group_keys=False).apply(process_group).reset_index(drop=True)
    df = df.sort_values(["encounter_reference", "effectiveDateTime"])

    # keep only last max_train_length points per encounter
    df = df.groupby("encounter_reference", group_keys=False).tail(max_train_length)

    counts = df["encounter_reference"].value_counts()
    valid_ids = counts[counts >= min_length].index
    out = df[df["encounter_reference"].isin(valid_ids)]

    logger.info("After filtering: %d rows across %d encounters (min_length=%d)", len(out), out["encounter_reference"].nunique(), min_length)
    return out


def compute_bootstrap_metrics(actual: np.ndarray, predicted: np.ndarray, n_bootstrap: int = 1000, ci_percentile: int = 95) -> Dict[str, Tuple[float, float, float]]:
    """Compute bootstrap confidence intervals for regression metrics."""
    mae_list, mse_list, rmse_list, mape_list, smape_list = [], [], [], [], []

    n = len(actual)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        a = actual[idx]
        p = predicted[idx]

        mae = np.mean(np.abs(a - p))
        mse = np.mean((a - p) ** 2)
        rmse = np.sqrt(mse)

        # Avoid divide-by-zero blowups
        denom = np.where(np.abs(a) == 0, np.nan, np.abs(a))
        mape = np.nanmean(np.abs(a - p) / denom) * 100

        smape_denom = (np.abs(a) + np.abs(p))
        smape_denom = np.where(smape_denom == 0, np.nan, smape_denom)
        smape = np.nanmean(np.abs(a - p) / smape_denom) * 100

        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mape_list.append(mape)
        smape_list.append(smape)

    def ci(values):
        return (
            float(np.nanmean(values)),
            float(np.nanpercentile(values, (100 - ci_percentile) / 2)),
            float(np.nanpercentile(values, 100 - (100 - ci_percentile) / 2)),
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


def autogluon_ci_prediction(
    model_name: str,
    output_path: str,
    df: pd.DataFrame,
    predictor: TimeSeriesPredictor,
    prediction_length: int,
    target_col: str,
    logger: logging.Logger,
    n_bootstrap: int = 1000,
) -> Dict[str, Tuple[float, float, float]]:
    """Generate predictions with confidence intervals and write metrics CSV."""
    logger.info("Forecasting with model=%s prediction_length=%d", model_name, prediction_length)

    df = df.copy()
    df["effectiveDateTime"] = pd.to_datetime(df["effectiveDateTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["effectiveDateTime"])
    df["effectiveDateTime"] = df["effectiveDateTime"].dt.tz_localize(None)

    # Split last N points for test set (per encounter)
    train_parts, test_parts = [], []
    for enc_id, g in df.groupby("encounter_reference"):
        test_rows = g.nlargest(prediction_length, "effectiveDateTime")
        train_rows = g.drop(test_rows.index)
        if len(train_rows) == 0 or len(test_rows) == 0:
            continue
        train_parts.append(train_rows)
        test_parts.append(test_rows)

    if not train_parts or not test_parts:
        raise ValueError("No usable train/test splits produced. Check time series length and prediction_length.")

    train_df = pd.concat(train_parts).sort_values(["encounter_reference", "effectiveDateTime"])
    test_df = pd.concat(test_parts).sort_values(["encounter_reference", "effectiveDateTime"])

    train_ts = TimeSeriesDataFrame.from_data_frame(train_df, id_column="encounter_reference", timestamp_column="effectiveDateTime")
    test_ts = TimeSeriesDataFrame.from_data_frame(test_df, id_column="encounter_reference", timestamp_column="effectiveDateTime")

    preds = predictor.predict(train_ts, model=model_name)[["mean", "0.1", "0.9"]]
    merged = test_ts.join(preds, how="inner")

    actual = merged[target_col].to_numpy().reshape(-1)
    predicted = merged["mean"].to_numpy().reshape(-1)

    if len(actual) == 0:
        raise ValueError("Merged test/prediction set is empty; cannot compute metrics.")

    metrics = compute_bootstrap_metrics(actual, predicted, n_bootstrap=n_bootstrap)

    # Write metrics to CSV
    out_df = pd.DataFrame(
        [{"Metric": m.upper(), "Estimate": v[0], "CI_Lower": v[1], "CI_Upper": v[2]} for m, v in metrics.items()]
    )
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("Saved metrics CSV: %s", output_path)

    return metrics


# -------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results", help="Directory for outputs + logs")
    parser.add_argument("--log_level", default="INFO", help="DEBUG|INFO|WARNING|ERROR|CRITICAL")
    parser.add_argument("--config_path", default="config_crp.yaml", help="Path to YAML config")
    args = parser.parse_args()

    logger = setup_logging(args.results_dir, args.log_level)

    logger.info("Starting CRP pipeline")
    logger.info("Working directory: %s", os.getcwd())
    logger.info("Results directory: %s", os.path.abspath(args.results_dir))

    try:
        config = load_config(args.config_path, logger)
        search = init_authentication(config, logger)
        fhir_count = str(config.get("fhir_count", 100))
        logger.info("FHIR _count parameter: %s", fhir_count)

        # -----------------------------------------------------------------
        # FHIR Queries
        # -----------------------------------------------------------------

        conditions_df = fetch_fhir_dataframe(
            search=search,
            df=pd.DataFrame(config["icd_codes"], columns=["icd-10"]),
            resource="Condition",
            params={"_count": fhir_count, "_sort": "_id", "recorded-date": "ge2026-01"},
            constraints={"code": "icd-10"},
            logger=logger,
            label="FHIR Query #1 Conditions",
        )

        if "encounter_reference" not in conditions_df.columns:
            raise KeyError("conditions_df is missing 'encounter_reference' column from FHIR conversion.")

        conditions_unique = (
            conditions_df.dropna(subset=["encounter_reference"])
            .drop_duplicates("encounter_reference")
        )
        logger.info("FHIR Query #1: unique encounters=%d", conditions_unique["encounter_reference"].nunique())

        encounters_df = fetch_fhir_dataframe(
            search=search,
            df=conditions_unique,
            resource="Encounter",
            params={"_count": fhir_count, "_sort": "_id"},
            constraints={"_id": "encounter_reference"},
            logger=logger,
            label="FHIR Query #2 Encounters",
        )

        if "id" in encounters_df.columns:
            logger.info("FHIR Query #2: encounters fetched=%d unique=%d", len(encounters_df), encounters_df["id"].nunique())
        else:
            logger.warning("encounters_df missing 'id' column; cannot compute unique encounter count reliably.")

        # Filter encounter class & diagnosis type if present
        stay_field = config["encounter_stay_type_field"][0]
        diag_field = config["encounter_diagnosis_type_field"][0]

        before = len(encounters_df)
        if stay_field in encounters_df.columns:
            # original script used 'class_code' explicitly
            if "class_code" in encounters_df.columns:
                encounters_df = encounters_df[encounters_df["class_code"].isin(config["encounter_stay_type_content"])]
                logger.info("Filtered encounters by class_code content; rows %d → %d", before, len(encounters_df))
            else:
                logger.warning("stay_field '%s' present but 'class_code' column not found; skipping stay filter", stay_field)
        else:
            logger.info("Stay filter field '%s' not present; skipping stay filter", stay_field)

        before = len(encounters_df)
        if diag_field in encounters_df.columns:
            encounters_df = encounters_df[encounters_df[diag_field].isin(config["encounter_diagnosis_type_content"])]
            logger.info("Filtered encounters by diagnosis field '%s'; rows %d → %d", diag_field, before, len(encounters_df))
        else:
            logger.info("Diagnosis filter field '%s' not present; skipping diagnosis filter", diag_field)

        # -----------------------------------------------------------------
        # Medication Filtering
        # -----------------------------------------------------------------

        meds_df = fetch_fhir_dataframe(
            search=search,
            df=pd.DataFrame(config["medication_codes"], columns=["atc_codes"]),
            resource="Medication",
            params={"_count": fhir_count, "_sort": "_id"},
            constraints={"code": "atc_codes"},
            logger=logger,
            label="FHIR Query #3 Medication",
        )
        if "id" not in meds_df.columns:
            raise KeyError("meds_df missing 'id' column")

        meds_df["medication_reference"] = "Medication/" + meds_df["id"].astype(str)
        logger.info("FHIR Query #3: medication references=%d", meds_df["medication_reference"].nunique())

        admin_df = fetch_fhir_dataframe(
            search=search,
            df=encounters_df,
            resource="MedicationAdministration",
            params={"_count": fhir_count, "_sort": "_id"},
            constraints={"context": "id"},
            logger=logger,
            label="FHIR Query #4 MedicationAdministration",
        )
        if "medicationReference_reference" not in admin_df.columns:
            raise KeyError("admin_df missing 'medicationReference_reference' column")

        before = len(admin_df)
        admin_df = admin_df[admin_df["medicationReference_reference"].isin(meds_df["medication_reference"])]
        logger.info("FHIR Query #4: filtered administrations %d → %d", before, len(admin_df))

        # -----------------------------------------------------------------
        # Patient Filtering
        # -----------------------------------------------------------------

        patients_df = fetch_fhir_dataframe(
            search=search,
            df=admin_df.drop_duplicates("subject_reference"),
            resource="Patient",
            params={"_count": fhir_count, "_sort": "_id"},
            constraints={"_id": "subject_reference"},
            logger=logger,
            label="FHIR Query #5 Patient",
        )

        birth_field = config["patients_birthdate_field"][0]
        encounters_with_age = compute_patient_age(encounters_df, patients_df, birth_field, logger)
        before = len(encounters_with_age)
        encounters_with_age = encounters_with_age[encounters_with_age["age"] >= 18]
        logger.info("Filtered to age>=18: rows %d → %d (unique encounters=%d)", before, len(encounters_with_age), encounters_with_age.get("id", pd.Series()).nunique())

        # -----------------------------------------------------------------
        # CRP Extraction
        # -----------------------------------------------------------------

        crp_df = fetch_fhir_dataframe(
            search=search,
            df=encounters_with_age,
            resource="Observation",
            params={"_count": fhir_count, "_sort": "date", "code": config["crp_laboratory_code"]},
            constraints={"encounter": "id"},
            logger=logger,
            label="FHIR Query #6 CRP Observation",
        )

        if "valueQuantity_value" not in crp_df.columns:
            raise KeyError("crp_df missing 'valueQuantity_value' column")

        if config["crp_unit"][0] == "mg/l":
            logger.info("Converting CRP unit mg/l by dividing by 10")
            crp_df["valueQuantity_value"] = crp_df["valueQuantity_value"] / 10.0

        logger.info(
            "FHIR Query #6: CRP rows=%d unique encounters with CRP=%s",
            len(crp_df),
            crp_df["encounter_reference"].nunique() if "encounter_reference" in crp_df.columns else "unknown",
        )

        # -----------------------------------------------------------------
        # Time-Series Preparation
        # -----------------------------------------------------------------

        logger.info("Loading AutoGluon predictor from %s", "crp_ensemble")
        predictor = TimeSeriesPredictor.load("crp_ensemble", require_version_match=False)
        logger.info("Loaded predictor. Models available: %s", predictor.model_names())

        resampled = resample_time_series(
            crp_df,
            rate="1d",
            min_length=4,
            impute="forward_fill",
            max_train_length=14,
            value_col="valueQuantity_value",
            logger=logger,
        )

        # -----------------------------------------------------------------
        # Forecasting
        # -----------------------------------------------------------------

        for model_name in predictor.model_names():
            output_path = os.path.join(args.results_dir, f"{model_name}_metrics.csv")
            logger.info("Running model: %s", model_name)

            _metrics = autogluon_ci_prediction(
                model_name=model_name,
                output_path=output_path,
                df=resampled,
                predictor=predictor,
                prediction_length=1,
                target_col="valueQuantity_value",
                logger=logger,
                n_bootstrap=1000,
            )

            logger.info("Finished model: %s (metrics saved)", model_name)

        logger.info("All done. Thank you. You are awesome.")
        return 0

    except Exception:
        # This logs the full stack trace to both console and run.log
        logger.exception("Pipeline failed with an unhandled exception")
        return 1


# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
