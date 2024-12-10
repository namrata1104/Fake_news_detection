import pandas as pd
import re
import string
import time

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from fnd.params import *

def basic_cleaning(data):
    # stripping:
    data['text'] = data['text'].str.strip()

    # tolower:
    data['text'] = data['text'].str.lower()

    # digit: Remove digits from each row of the ‘text’ column
    data['text'] = data['text'].apply(lambda x: ''.join(char for char in x if not char.isdigit()))

    # punctuation: Remove all punctuation marks from the ‘text’ column
    data['text'] = data['text'].str.replace(r'[{}]'.format(re.escape(string.punctuation)), '', regex=True)

    # delete html-tags
    data['text'] = data['text'].apply(lambda x: re.sub('<[^<]+?>', '', x))

    return data['text']

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
    df.drop_duplicates(inplace=True)
    df.dropna(how='any', axis=0, inplace=True)

    ##########################################################
    #                    basic cleaning                      #
    ##########################################################
    s = time.time()
    df['text'] = basic_cleaning(df['text'])
    time_to_clean = time.time() - s
    print('Time for basic cleaning {:.2f} s'.format(time_to_clean))

    print("✅ data cleaned")

    return df

def get_data_with_cache(
        gcp_project: str = {GCP_PROJECT},
        query: str = "",
        cache_path: Path = Path("../raw_data/WELFake_Dataset.csv"),
        data_has_header: bool = True
    ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else: # Block not used yet
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
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
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
