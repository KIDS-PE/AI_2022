# Development and Verification of a Time Series AI Model for Acute Kidney Injury Detection Based on a Multicenter Distributed Research Network

## Purpose
The purpose of this project is to develop and validate a drug-induced acute kidney injury (AKI) prediction model using the Common Data Model (CDM). The model is designed to detect AKI using time series data from a multicenter distributed research network.

## Description
This repository contains the following scripts:

### scripts
- 1_create_cohort_person_in_db.ipynb: Creates a cohort of patients with AKI and non-AKI.
- 2_propensity_score_matching.ipynb: Matches AKI and non-AKI patients using propensity score matching.
- 3_merge_domain_data.ipynb: combines domain data in the CDM.
- 4_feature_selections.ipynb: Performs feature selection for the model.
- 7_preprocessing_lstm.ipynb: Preprocesses data for the time series model.
- 8_imv_lstm_attention.ipynb: Trains and evaluates the time series model with attention mechanism.

### scripts2
- for extract demographic data and analysis result

### Getting Started

To get started with the project, follow these steps:

1. Install project-related requirements in Python (if necessary, create a virtual environment) by running the following command:
bash
'''
pip install -r requirements.txt
pip install psycopg2-binary
'''
2. Edit the config.json file with the appropriate parameters:
- "working_date": Date to run the program.
- "dbms": MSSQL or PostgreSQL.
- @Server, @user, @password, and @port : database connection information.
- @database : cdm database name
- @cdm_database_schema : cdm database schema that contains standardized data tables (person, drug_exposure, etc..)
- @person_database_schema : extra schema to save tables during analysis. (NOTE that the user must have permissions to read, write, remove tables in the schema.)
- @target_database_schema : same as @person_database_schema
- @vocabulary_database_schema : cdm database schema that contains standardized vocabularies tables (concept, concept_ancestor, concept_relationship, etc...)
- meas: meas_concept_id of AST, ALT, ALP, Total Bilirubin, Creatinine, PT(INR) (used by the institution).
- translatequerysql0 ~ translatequerysql4: modify @cdm_database_schema, @target_database_schema, @vocabulary_database_schema as described above

3. Execute the Jupyter Notebook files in the scripts folder in the order of folder number. (Script 1 > 2 > 3 > 4 > 7 > 8)

### Results Export
The following directories are generated during the program execution:

- data: Data for model training.
- result: Preprocessing results and evaluation of learning performance.
To export the results, compress and export the result directory.

## Authors
- This project was developed by a researcher at the DHLab, Department of Biomedical Systems Informatics, Yonsei University College of Medicine, Seoul.
- This project was partially modified by Office of Pharmacoepidemiology and Big Data, Korea Institute of Drug Safety & Risk Management.

## Version History
- v1.0.0 : Initial Release

## Acknowledgments
This study was supported by a grant from the Korea Institute of Drug Safety and Risk Management in 2021.

## Citation
preprint (in preparation)
