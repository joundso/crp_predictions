#!/usr/bin/env python
# coding: utf-8

# In[2]:



from fhir_pyrate import Ahoy
from fhir_pyrate import Pirate
import os
import pandas as pd
import yaml
from dotenv import load_dotenv
from os.path import join, dirname
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import os
import pandas as pd
import autogluon
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import argparse
import os

print("Running in directory:", os.getcwd())
print("Files in /app:", os.listdir("/app"))
print("Is predictor.pkl there?:", os.path.exists("crp_ensemble/predictor.pkl"))


# Parse base paths
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", default="./results", help="Path to results folder")
args = parser.parse_args()


RESULTS_DIR = args.results_dir


pd.set_option('max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Load the YAML config file
with open('config_crp.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load configs
atc_codes = config['medication_codes']
icd_codes = config['icd_codes']
#loinc_category_laboratory = config['loinc_category_laboratory']
crp_laboratory_code = config['crp_laboratory_code']
fhir_extract = config['fhir_extract']
models_eval = config['models_eval']
diagnosis_type_ref = config['diagnosis_type_ref']
authentication_method = config['authentication_method']
authentication_method = authentication_method[0]

conditions = pd.DataFrame(icd_codes, columns=['icd-10'])
medications = pd.DataFrame(atc_codes, columns=['atc_codes'])
crp_unit = config['crp_unit']
crp_unit = crp_unit[0]
encounter_diagnosis_type_field = config['encounter_diagnosis_type_field']
encounter_diagnosis_type_field = encounter_diagnosis_type_field[0]
encounter_diagnosis_type_content = config['encounter_diagnosis_type_content']
encounter_stay_type_field = config['encounter_stay_type_field']
encounter_stay_type_field = encounter_stay_type_field[0]
encounter_stay_type_content = config['encounter_stay_type_content']
patients_birthdate_field = config['patients_birthdate_field']
patients_birthdate_field = patients_birthdate_field[0]
fhir_count = config.get("fhir_count", 100)   # default to 1000 if missing
fhir_count = str(fhir_count)  

load_dotenv(dotenv_path="/app/env_py.env")  # or relative: load_dotenv(".env")



if authentication_method == 'basic_auth':
    FHIR_USER = os.environ["FHIR_USER"]
    FHIR_SERVER_URL = os.environ["FHIR_SERVER_URL"]
    auth = Ahoy(
    auth_url=FHIR_SERVER_URL,
    auth_type="BasicAuth",
    username=FHIR_USER,
    auth_method="env",
    )
    search = Pirate(
    FHIR_SERVER_URL,
    auth=auth,
    print_request_url=False,
    )
    
else:
    BASIC_AUTH = os.environ["BASIC_AUTH"]
    REFRESH_AUTH = os.environ["REFRESH_AUTH"]
    SEARCH_URL = os.environ["SEARCH_URL"]
    auth = Ahoy(
    auth_type="token",
    auth_method="env",
    auth_url=BASIC_AUTH,
    refresh_url=REFRESH_AUTH,
    )
    
    search = Pirate(
    auth=auth,
    base_url=SEARCH_URL, # e.g. "http://hapi.fhir.org/baseDstu2"
    print_request_url=False, # If set to true, you will see all requests
    )

#### Start with all cancer conditions (and admission diagnosis)
# Search FHIR-resource Condition, filtered by defined ICD-10 codes and date >= 2015-01
if diagnosis_type_ref == ['condition']:
    dr_bundles = search.trade_rows_for_bundles(
        conditions,
        resource_type="Condition",
        #request_params={"_count": fhir_count,"_sort":'_id',"recorded-date": "ge2025-01"},
        request_params={"_count": fhir_count,"_sort":'_id',"recorded-date": "ge2022-01","category":"ADM"},
        df_constraints={"code": "icd-10"})
    conditions_df = search.bundles_to_dataframe(bundles=dr_bundles)
    print ('FHIR Query #1: Number of unique encounters with filtered ICD-10 codes: ', conditions_df['encounter_reference'].nunique())
else:
    dr_bundles = search.trade_rows_for_bundles(
        conditions,
        resource_type="Condition",
        request_params={"_count": fhir_count,"_sort":'_id',"recorded-date": "ge2022-01"},
        #request_params={"_count": fhir_count,"_sort":'_id',"recorded-date": "ge2025-01","category":"ENT"},
        df_constraints={"code": "icd-10"})
    conditions_df = search.bundles_to_dataframe(bundles=dr_bundles)
    print ('FHIR Query #1 (Conditions): Number of unique encounters in conditions with filtered ICD-10 codes: ', conditions_df['encounter_reference'].nunique())

conditions_df_unique = conditions_df.dropna(subset=['encounter_reference']).drop_duplicates(subset=['encounter_reference'])
#conditions_df_unique = conditions_df.drop_duplicates(subset=['encounter_reference'])
#conditions_df_unique.to_csv('conditions_df_adm_unique.csv', sep = ';', index=False)


# In[10]:


#### Restrict encounters with all cancer conditions to inpatients
dr_bundles = search.trade_rows_for_bundles(
    conditions_df_unique,
    resource_type="Encounter",
    #request_params={"_count": fhir_count,"_sort":'_id',"recorded-date": "ge2025-01"},
    request_params={"_count": fhir_count,"_sort":'_id'},
    df_constraints={"_id": "encounter_reference"}
  )
encounters_df = search.bundles_to_dataframe(bundles=dr_bundles)
print ('FHIR Query #2 (Encounters): Number of unique encounters filtered by conditions: ', encounters_df['id'].nunique())
#encounters_df.to_csv('encounters_df.csv', sep = ';', index=False)


# In[11]:


# Check if 'class_code' column exists and filter if it does
if encounter_stay_type_field in encounters_df.columns:
    encounters_df = encounters_df[encounters_df['class_code'].isin(encounter_stay_type_content)]

# Check if 'diagnosis_use_coding_code' column exists and filter if it does
if encounter_diagnosis_type_field in encounters_df.columns:
    encounters_df = encounters_df[encounters_df[encounter_diagnosis_type_field].isin(encounter_diagnosis_type_content)]
                                                          
print ('Filter FHIR Query #2: Only ',encounter_stay_type_content,' encounters with ', encounter_diagnosis_type_content, ' : ', encounters_df['id'].nunique())
                                                             


# In[12]:


#### Get all medication reference IDs for antibiotic ATC codes
dr_bundles = search.trade_rows_for_bundles(
    medications,
    resource_type="Medication",
    request_params={"_count": fhir_count, "_sort": "_id"},
    df_constraints={"code": "atc_codes"}
)
      # Convert the returned bundles to a dataframe
medications_df = search.bundles_to_dataframe(bundles=dr_bundles)
medications_df['medication_reference'] = 'Medication/' + medications_df['id'].astype(str)
print ('FHIR Query #3: Number of medication references with filtered ATC codes: ', medications_df['id'].nunique())

#medications_df.to_csv('medication_references_df.csv', sep = ';', index=False)


# In[13]:


# Filter conditions_df by encounters in which i.v. antibiotic medications were administered
dr_bundles = search.trade_rows_for_bundles(
    encounters_df,
    resource_type="MedicationAdministration",
    request_params={"_count": fhir_count, "_sort": "_id"},
    df_constraints={"context": "id"})
      # Convert the returned bundles to a dataframe
medications_admin_df = search.bundles_to_dataframe(bundles=dr_bundles)
print ('FHIR Query #4: Number of medication administrations during filtered encounters: ', medications_admin_df['id'].nunique())

#medications_admin_df.to_csv('medication_admin_df.csv', sep = ';', index=False)


# In[14]:


medications_admin_df = medications_admin_df[
    medications_admin_df['medicationReference_reference'].isin(medications_df['medication_reference'])
]
print ('Filter FHIR Query #4: Filter medication administrations for ATC code references in medications: ', medications_admin_df['id'].nunique())


# In[16]:


medications_admin_df_unique_subject = medications_admin_df.drop_duplicates(subset=['subject_reference'])
dr_bundles = search.trade_rows_for_bundles(
    medications_admin_df_unique_subject,
    resource_type="Patient",
    request_params={"_count": fhir_count, "_sort": "_id"},
    df_constraints={"_id": "subject_reference"},
  )
patients_df = search.bundles_to_dataframe(
      bundles=dr_bundles,fhir_paths=["id", patients_birthdate_field])
print ('FHIR Query #5: Number of unique patients receiving antibiotics during filtered encounters: ', patients_df['id'].nunique())


# In[17]:


# Exclude patients younger than 18
patients_df[patients_birthdate_field] = pd.to_datetime(patients_df[patients_birthdate_field], errors='coerce')
encounters_df['period_start'] = pd.to_datetime(encounters_df['period_start'], utc=True)
encounters_df['period_start'] = encounters_df['period_start'].dt.tz_localize(None)
encounters_df['subject_identifier'] = encounters_df['subject_reference'].str.replace('Patient/', '')

# Merge the two dataframes based on a common patient identifier (assuming 'patient_id' exists in both)
merged_df = pd.merge(encounters_df, patients_df[['id', patients_birthdate_field]], left_on='subject_identifier', right_on='id', how='left')
# Calculate age by subtracting the birthDate from the recordedDate and dividing by 365.25 to get the number of years
merged_df['age'] = (merged_df['period_start'] - merged_df[patients_birthdate_field]).dt.days // 365

# Now, if you want to update the original conditions_df_filtered_unqiue_patients DataFrame:
encounters_df = encounters_df.merge(
    merged_df[['subject_identifier', 'period_start', 'age']], 
    on=['subject_identifier', 'period_start'], 
    how='left'
)
encounters_df = encounters_df[encounters_df['age'] >= 18]
print ('Filter FHIR Query #5: Filter encounters for patients >= 18 at start of encounter: ', encounters_df['id'].nunique())



# In[18]:


# Filter CRP values
dr_bundles = search.trade_rows_for_bundles(
  encounters_df,
  resource_type="Observation",
  #request_params={"category": "26436-6","_sort":"date","_count": fhir_count},
  request_params={"_count": fhir_count,"_sort":"date", "code": crp_laboratory_code},
  df_constraints={"encounter": "id"}
)
# Convert the returned bundles to a dataframe
crp_df = search.bundles_to_dataframe(bundles=dr_bundles)
if crp_unit == 'mg/l':
    crp_df['valueQuantity_value'] = crp_df['valueQuantity_value'] / 10
print ('FHIR Query #6: Number of CRP values for patients receiving antibiotics during filtered encounters: ', crp_df['id'].nunique())
print ('Number of unique encounters with at least one CRP value: ', crp_df['encounter_reference'].nunique())
#crp_df.to_csv('crp_df.csv', sep = ';', index=False)
#crp_df = pd.read_csv('crp_df.csv',sep=';')





# In[35]:


def autogluon_data_ci_regression(model_name, output_dir_forecasting, forecasting_filename,
                                  resampled_df, predictor, prediction_length, value_to_predict, n_bootstrap=1000):
    # Prepare recorded_time column
    resampled_df['effectiveDateTime'] = resampled_df['effectiveDateTime'].dt.tz_localize(None)

    # Build TimeSeriesDataFrame
    train_data_covar = TimeSeriesDataFrame.from_data_frame(
        resampled_df,
        id_column="encounter_reference",
        timestamp_column="effectiveDateTime"
    )

    # Train/test split
    def split_train_test(group):
        test_rows = group.nlargest(prediction_length, 'effectiveDateTime')
        train_rows = group.drop(test_rows.index)
        return train_rows, test_rows

    train_list = []
    test_list = []
    for name, group in resampled_df.groupby('encounter_reference'):
        train_rows, test_rows = split_train_test(group)
        train_list.append(train_rows)
        test_list.append(test_rows)

    train_data = pd.concat(train_list).reset_index(drop=True).sort_values(by=['encounter_reference', 'effectiveDateTime'])
    test_data = pd.concat(test_list).reset_index(drop=True).sort_values(by=['encounter_reference', 'effectiveDateTime'])

    train_data_ = train_data[['encounter_reference', 'effectiveDateTime', value_to_predict]].copy()
    train_data_['effectiveDateTime'] = train_data_['effectiveDateTime'].dt.tz_localize(None)

    test_data_ = test_data[['encounter_reference', 'effectiveDateTime', value_to_predict]].copy()
    test_data_['effectiveDateTime'] = test_data_['effectiveDateTime'].dt.tz_localize(None)

    # Build TimeSeriesDataFrames
    #train_data_.info()
    train_data = TimeSeriesDataFrame.from_data_frame(train_data_, id_column="encounter_reference", timestamp_column="effectiveDateTime")
    test_data = TimeSeriesDataFrame.from_data_frame(test_data_, id_column="encounter_reference", timestamp_column="effectiveDateTime")


    # Make predictions
    predictions = predictor.predict(train_data, model=model_name)
    predictions_ = predictions[['mean', '0.1', '0.9']]

    # Merge predictions with actuals
    merged_data = test_data.join(predictions_, how="inner")
    actuals = merged_data[value_to_predict].values.reshape(-1, 1)
    predictions = merged_data['mean'].values.reshape(-1, 1)

    # Bootstrapping regression metrics
    metrics_forecasting = bootstrap_metrics_auto(actuals, predictions, n_bootstrap=n_bootstrap)

    # Format forecasting metrics
    forecasting_metrics_data = []
    for metric, values in metrics_forecasting.items():
        forecasting_metrics_data.append({
            "Metric": metric.upper(),
            "Point Estimate": values[0],
            "95% CI Lower": values[1],
            "95% CI Upper": values[2]
        })

    # Save forecasting metrics
    forecasting_metrics_df = pd.DataFrame(forecasting_metrics_data)
    os.makedirs(output_dir_forecasting, exist_ok=True)
    forecasting_csv_path = os.path.join(output_dir_forecasting, forecasting_filename)
    forecasting_metrics_df.to_csv(forecasting_csv_path, index=False)
    print(f"Forecasting metrics saved to {forecasting_csv_path}")
    
    merged_df_reset = merged_data.reset_index()
    print(merged_df_reset.columns)  # Debugging: inspect what columns are actually present


    # Save actual vs predicted values
    predictions_output_df = merged_df_reset[[
        #'item_id',
        #'timestamp',
        value_to_predict,
        'mean',
        '0.1',
        '0.9'
    ]]
    predictions_output_df.rename(columns={
        value_to_predict: 'actual',
        'mean': 'predicted',
        '0.1': 'ci_lower_0.1',
        '0.9': 'ci_upper_0.9'
    }, inplace=True)

    predictions_csv_path = os.path.join(output_dir_forecasting, f"{model_name}_actual_vs_predicted.csv")
    #predictions_output_df.to_csv(predictions_csv_path, index=False)
    print(f"Actual vs. predicted values saved to {predictions_csv_path}")
    
    return predictions, actuals, train_data, test_data, merged_data


# In[15]:


def resample_ts(df,resample_rate,min_ts_length, imputation,max_train_length,value_to_predict):
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'],utc=True)
    df = df.dropna(subset=['effectiveDateTime'])

    # Define the aggregation functions for numerical columns
    #print(df.info())
    agg_funcs = {
            value_to_predict: 'max'
    }
    
    # Define a function to resample, aggregate, and interpolate within each group
    def resample_aggregate_and_interpolate(group):
        group = group.set_index('effectiveDateTime')
        resampled = group.resample(resample_rate).agg(agg_funcs)
        resampled['encounter_reference'] = group['encounter_reference'].iloc[0]  # Add encounter_id back
        resampled = resampled.reset_index()
        
        if imputation == 'interpolate':
            # Interpolate missing values linearly within each group
            resampled = resampled.groupby('encounter_reference').apply(
                lambda x: x.interpolate(method='linear', limit_direction='forward')
            ).reset_index(drop=True)
        elif imputation == 'forward_fill':
                # Forward fill missing values within each group
            resampled = resampled.groupby('encounter_reference').apply(
                lambda x: x.ffill()
            ).reset_index(drop=True)
            resampled = resampled.groupby('encounter_reference').apply(
                lambda x: x.bfill()
            ).reset_index(drop=True)
            resampled = resampled.fillna(0)
        else:
            pass

        return resampled
    
    # Apply the function to each group and combine results
    resampled_df = df.groupby('encounter_reference').apply(resample_aggregate_and_interpolate).reset_index(drop=True)

    data_sorted = resampled_df.sort_values(by=['encounter_reference', 'effectiveDateTime'])
    
    # Group by 'encounter_id' and get the last 10 rows for each group
    data_recent = data_sorted.groupby('encounter_reference').tail(max_train_length)

    # Reset index if needed
    resampled_df = data_recent.reset_index(drop=True)
    resampled_df = resampled_df[resampled_df[value_to_predict].notna()]

    #resampled_df = resampled_df.dropna(subset=['crp'])
    # Filter out encounter_ids with less than n data points
    valid_encounters = resampled_df['encounter_reference'].value_counts()
    valid_encounters = valid_encounters[valid_encounters >= min_ts_length].index
    resampled_df = resampled_df[resampled_df['encounter_reference'].isin(valid_encounters)]
    # Calculate mean and standard deviation of imputations
    #imputation_values = list(imputation_counts.values())
    #mean_imputations = np.mean(imputation_values)
    #std_imputations = np.std(imputation_values)

    # Print the statistics
    #print(f"Mean number of imputations per encounter_id: {mean_imputations}")
    #print(f"Standard deviation of imputations per encounter_id: {std_imputations}")
    return resampled_df


# In[37]:


def bootstrap_metrics_auto(actual_values, predicted_values, n_bootstrap=1000, ci_percentile=95):
    # Initialize arrays to store metrics across bootstraps
    mae_bootstrap = []
    mse_bootstrap = []
    rmse_bootstrap = []
    mape_bootstrap = []
    smape_bootstrap = []

    n_samples = len(actual_values)

    for _ in range(n_bootstrap):
        # Sample with replacement
        resample_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        actual_resample = np.array([actual_values[i] for i in resample_indices])
        predicted_resample = np.array([predicted_values[i] for i in resample_indices])

        # Calculate metrics for this bootstrap sample
        mae = np.mean(np.abs(actual_resample - predicted_resample))
        mse = np.mean((actual_resample - predicted_resample) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_resample - predicted_resample) / actual_resample)) * 100
        smape = np.mean(np.abs(actual_resample - predicted_resample) / (np.abs(actual_resample) + np.abs(predicted_resample))) * 100

        # Append the results
        mae_bootstrap.append(mae)
        mse_bootstrap.append(mse)
        rmse_bootstrap.append(rmse)
        mape_bootstrap.append(mape)
        smape_bootstrap.append(smape)

    # Calculate mean and confidence intervals
    metrics = {
        "mae": (np.mean(mae_bootstrap), np.percentile(mae_bootstrap, (100 - ci_percentile) / 2), np.percentile(mae_bootstrap, 100 - (100 - ci_percentile) / 2)),
        "mse": (np.mean(mse_bootstrap), np.percentile(mse_bootstrap, (100 - ci_percentile) / 2), np.percentile(mse_bootstrap, 100 - (100 - ci_percentile) / 2)),
        "rmse": (np.mean(rmse_bootstrap), np.percentile(rmse_bootstrap, (100 - ci_percentile) / 2), np.percentile(rmse_bootstrap, 100 - (100 - ci_percentile) / 2)),
        "mape": (np.mean(mape_bootstrap), np.percentile(mape_bootstrap, (100 - ci_percentile) / 2), np.percentile(mape_bootstrap, 100 - (100 - ci_percentile) / 2)),
        "smape": (np.mean(smape_bootstrap), np.percentile(smape_bootstrap, (100 - ci_percentile) / 2), np.percentile(smape_bootstrap, 100 - (100 - ci_percentile) / 2)),
    }

    return metrics



######## Start the predictions here
predictor = TimeSeriesPredictor.load("crp_ensemble",require_version_match=False)




resample_rate = '1d'
min_ts_length = 4
imputation = 'forward_fill'
max_train_length = 14
value_to_predict = 'valueQuantity_value'
prediction_length = 1
output_dir_forecasting = RESULTS_DIR
resampled_df = resample_ts(crp_df,resample_rate,min_ts_length,imputation,max_train_length,value_to_predict)


for model_name in predictor.model_names():
    print (model_name)
    predictions, actuals, train_data, test_data, merged_data = autogluon_data_ci_regression(model_name,output_dir_forecasting,model_name,resampled_df, predictor,prediction_length,value_to_predict)

print('All done. Thank you. You are awesome.')


