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
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """
    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
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
        print(f"Dataset {BQ_DATASET} was been created.")

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

def get_processed_data():
    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # Below, our columns are called [text, label] on BQ
    # Query PROCESSED data from BigQuery using `get_data_with_cache`
    query = f"""SELECT {",".join(COLUMN_NAMES_RAW)} FROM {GCP_PROJECT}.{BQ_DATASET}.{DATA_PROCESSED_NAME}"""

    # Retrieve data using `get_data_with_cache`
    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"{DATA_PROCESSED_NAME}.csv")
    data_processed = get_data_with_cache(
        gcp_project = GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )
    return data_processed

from typing import Tuple
import pandas as pd
import numpy as np

def split_data(data: pd.DataFrame, split_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits and shuffles the processed data into training and testing sets.

    Args:
        data (pd.DataFrame): The processed dataset.
        split_ratio (float): The ratio for splitting the dataset into training and testing sets (e.g., 0.2 for 20% test data).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train (features for training)
            - y_train (labels for training)
            - X_test (features for testing)
            - y_test (labels for testing)
    """

    # Calculate the length of the training set
    train_length = int(len(data) * (1 - split_ratio))

    # Split the data into training and testing sets
    data_train = data[:train_length]
    data_test = data[train_length:]

    # Separate features (X) and labels (y)
    X_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    X_test = data_test[:, :-1]
    y_test = data_test[:, -1]

    return X_train, y_train, X_test, y_test
