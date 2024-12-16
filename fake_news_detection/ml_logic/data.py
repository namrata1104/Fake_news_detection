import pandas as pd
import time

from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from colorama import Fore, Style
from pathlib import Path

from fake_news_detection.params import *

def get_data_with_cache(
        gcp_project: str = {GCP_PROJECT},
        query: str = "",
        cache_path: Path = "",
        data_has_header: bool = True
    ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)

        # load raw data to bq
        #upload_data_to_bq(df, gcp_project=GCP_PROJECT, bq_dataset=BQ_DATASET, table=f'{DATA_RAW_NAME}', truncate=True)
        print(f"✅ Data loaded from disk and stored in BigQuery DB, with shape {df.shape}")

    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=GCP_PROJECT, location=BQ_REGION)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()
        print(f"✅ Data loaded from BigQuery DB, with shape {df.shape}")
        # Store as CSV localy if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)
            print(f"✅ loaded data from the disk stored in BigQuery DB, with shape {df.shape}")

    print(f"✅ Data loaded, with shape {df.shape} is done!")

    return df

def upload_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    # TODO delete return

    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """
    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{GCP_PROJECT}.{BQ_DATASET}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data into full_table_name
    client = bigquery.Client(project=GCP_PROJECT, location=BQ_REGION)
    # Check if the dataset exists
    try:
        # Attempt to load the dataset
        dataset = client.get_dataset(f"{GCP_PROJECT}.{BQ_DATASET}")
    except NotFound:
        # If the dataset is not found, create it
        print(f"Dataset {BQ_DATASET} not found. Creating it...")
        dataset = bigquery.Dataset(f"{GCP_PROJECT}.{BQ_DATASET}")
        dataset = client.create_dataset(dataset)
        _wait_for_dataset(client)
        print(f"Dataset {BQ_DATASET} has been created.")

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data from dataframe into BigQuery DB
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

# Function to wait until the dataset exists
def _wait_for_dataset(client, timeout=6):
    """Wait until the dataset exists, or timeout after a given period."""
    start_time = time.time()
    while True:
        try:
            client.get_dataset(BQ_DATASET)
            print(f"Dataset {BQ_DATASET} is now available.")
            break
        except NotFound:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Dataset {BQ_DATASET} not available after {timeout} seconds.")
            time.sleep(2)  # Check every 2 seconds
