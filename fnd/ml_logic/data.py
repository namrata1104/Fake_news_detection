import pandas as pd
import re
import string
import time

from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from colorama import Fore, Style
from pathlib import Path

from fnd.params import *

def basic_cleaning(df):
    # stripping:
    df['text'] = df['text'].str.strip()

    # tolower:
    df['text'] = df['text'].str.lower()

    # digit: Remove digits from each row of the ‘text’ column
    df['text'] = df['text'].apply(lambda x: ''.join(char for char in x if not char.isdigit())).astype('string')

    # punctuation: Remove all punctuation marks from the ‘text’ column
    df['text'] = df['text'].str.replace(r'[{}]'.format(re.escape(string.punctuation)), '', regex=True)

    # delete html-tags
    df['text'] = df['text'].apply(lambda x: re.sub('<[^<]+?>', '', x)).astype('string')

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    ##########################################################
    #                 Delete redundant column                #
    ##########################################################
    # delete redundant column 'Unnamed: 0'
    df.drop('Unnamed: 0', axis=1, inplace=True)

    ##########################################################
    #               Compressing Data                         #
    ##########################################################
    # Compressing datatypes: raw_data by setting types to DTYPES_RAW
    df = df.astype(DTYPES_RAW)
    ##########################################################
    #        Handling missing and redundant Data             #
    ##########################################################
    # reduce column title after combining
    # filling the missing data with spaces
    df = df.fillna(' ') ## aplying to na via fillna
    # combine the text and the title
    df['text'] = df['title'] + df['text']
    # delete title column, beacuse it's included in text column
    df.drop('title',axis=1, inplace=True)

    # Remove buggy transactions
    df.drop_duplicates(inplace=True)
    ##########################################################
    #                     check Balancing                    #
    ##########################################################
    # Calculate percentage of NaN values in each column
    # Balancing: Calculation of the distribution in per cent
    label_counts = df['label'].value_counts(normalize=True) * 100
    # Formatted output
    print(f"Balancing result is label 0: {label_counts.get(0, 0):.2f}%, label 1: {label_counts.get(1, 0):.2f}%")

    ##########################################################
    #                     NaN Handling                       #
    ##########################################################
    # Calculate percentage of NaN values in each column
    nan_percentage = (df.isna().sum() / len(df)) * 100
    # Format output with percentage symbol
    formatted_nan_percentage = nan_percentage.apply(lambda x: f"{x:.2f}%")
    # Print each column's NaN percentage
    for col, percentage in formatted_nan_percentage.items():
        print(f"{col}: {percentage}")
    # Remove buggy transactions
    df.dropna(how='any', axis=0, inplace=True)
    ##########################################################
    #                    basic cleaning                      #
    ##########################################################
    s = time.time()
    df = basic_cleaning(df)
    print(df.info())
    time_to_clean = time.time() - s
    print('Time for basic cleaning {:.2f} s'.format(time_to_clean))

    print("✅ data cleaned")

    return df

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
        load_data_to_bq(df, gcp_project=GCP_PROJECT, bq_dataset=BQ_DATASET, table=f'{DATA_RAW_NAME}', truncate=True)

    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=GCP_PROJECT, location=BQ_REGION)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV localy if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

""" not used yet """
def load_data_to_bq(
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

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    # TODO: simplify this solution if possible, but students may very well choose another way to do it
    # We don't test directly against their own BQ tables, but only the result of their query
    #data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

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
        wait_for_dataset(client)
        print(f"Dataset {BQ_DATASET} has been created.")

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

# Function to wait until the dataset exists
def wait_for_dataset(client, timeout=6):
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
