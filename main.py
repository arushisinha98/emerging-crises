"""
Main data preparation pipeline for financial crisis prediction.
This script orchestrates the complete data preprocessing workflow including:

1. Loading World Bank, OECD, Yahoo Finance, and JST data
(creates the following files:
    A. data/worldbank_variables.csv
    B. data/oecd_variables.csv
    C. data/yahoofinance_variables.csv
    D. data/jst_variables.csv)
(requires the following files:
    A. data/JSTdatasetR6.xlsx)

2. Creating crisis labels from historical crisis data
(requires the following files:
    A. data/a-new-comprehensive-database-of-financial-crises.xlsx
    B. data/20160923_global_crisis_data.xlsx)

3. Uploading ISO country identifiers
(requires the following file: data/iso-standard-master.csv)

Set your Hugging Face Hub username and token in .env and run this script to generate the
relevant datasets to reproduce the experiment results.
"""

import pandas as pd
import os
import dotenv
dotenv.load_dotenv()
username = os.getenv("HUGGINGFACE_USERNAME")

#################################################################
## LOAD DATA ##
#################################################################
def LOAD_DATA():
    from src.data.loader import WorldBankDataLoader, OECDDataLoader, YahooFinanceDataLoader, JSTDataLoader

    ## Load World Bank data
    loader = WorldBankDataLoader(config_path = 'src/config.json')
    developed, emerging = loader.run_data_pipeline()
    print("World Bank Data Download completed successfully.")
    print(f"Dataframe has size: Developed={developed.shape}, Emerging={emerging.shape}")

    ## Load OECD data
    loader = OECDDataLoader(config_path = 'src/config.json')
    developed, emerging = loader.run_data_pipeline()
    print("OECD Data Download completed successfully.")
    print(f"Dataframe has size: Developed={developed.shape}, Emerging={emerging.shape}")

    ## Load Yahoo Finance data
    loader = YahooFinanceDataLoader(config_path = 'src/config.json')
    data = loader.run_data_pipeline()
    print("Yahoo Finance Data Download completed successfully.")
    print(f"Dataframe has size {data.shape}")

    ## Load JST data
    loader = JSTDataLoader(config_path = 'src/config.json')
    data = loader.run_data_pipeline()
    print("JST Data Download completed successfully.")
    print(f"Dataframe has size {data.shape}")

#################################################################
## CREATE CRISIS LABELS ##
#################################################################
def CREATE_CRISIS_LABELS():
    from src.data.loader import CrisisLabeller
    from src.data.data_utilities import upload_to_huggingface

    ## Create crisis labels
    labeller1 = CrisisLabeller(
        filepath = "data/a-new-comprehensive-database-of-financial-crises.xlsx",
        sheetname = "Crisis",
    )
    labels1 = labeller1.create_labels(lookback_years=2, recovery_years=0)
    labeller2 = CrisisLabeller(
        filepath = "data/20160923_global_crisis_data.xlsx",
        sheetname = "Sheet1"
    )
    labels2 = labeller2.create_labels(lookback_years=2, recovery_years=0)
    labels = pd.concat([labels1, labels2], axis=0).drop_duplicates()
    labels = labels.set_index('Country')
    upload_to_huggingface(labels, repo_name="crisis-labels-dataset", config_name="train")

    print("Crisis Labelling completed successfully.")
    print(f"Dataframe has size {labels.shape}")

#################################################################
## UPLOAD ISO STANDARD ##
#################################################################
def UPLOAD_ISO():
    from src.data.data_utilities import upload_to_huggingface

    ## Upload ISO standard dataset
    df = pd.read_csv('data/iso-standard-master.csv')
    df = df.set_index('alpha-3')
    upload_to_huggingface(df, repo_name="iso-standard-master", config_name='train')

if __name__ == "__main__":
    
    LOAD_DATA()
    CREATE_CRISIS_LABELS()
    UPLOAD_ISO()