from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import time
import numpy as np
import pandas as pd
import re
import string
from fake_news_detection.params import *
from fake_news_detection.ml_logic.data import get_data_with_cache, upload_data_to_bq
from pathlib import Path
from colorama import Fore, Style

def preprocess() -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query raw data from BigQuery using `get_data_with_cache`
    query = f"""SELECT {",".join(COLUMN_NAMES_RAW)} FROM {GCP_PROJECT}.{BQ_DATASET}.{DATA_RAW_NAME}"""

    # Retrieve data using `get_data_with_cache`
    df_query_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"{DATA_RAW_NAME}.csv")
    df = get_data_with_cache(query = query, gcp_project = {GCP_PROJECT},
                                cache_path = df_query_cache_path, data_has_header = True)

    # can be used to reduce runing time
    df = df.head(int(DATA_SIZE))
    print(df.shape)

    # cleaning data
    df = prepare_basic_clean_data(df)

    X = df.drop('label', axis=1)
    y = df['label']

    # start preprocessing data for nlp
    print('preprocessing data start ...')
    s = time.time()
    X_processed = X['text'].apply(preprocess_feature)
    print(f"Time to preprocessing data : {time.time() - s:.2f} seconds")

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()
    X_processed = X_processed.to_numpy().reshape(-1, 1)  # Form wird zu (100, 1)


    y = y.to_numpy().reshape(-1, 1)  # Form wird ebenfalls zu (100, 1)


    data_processed = pd.DataFrame(np.concatenate((X_processed, y), axis=1), columns=COLUMN_NAMES_RAW)
    # Store as CSV localy if at least one valid line is processed
    if data_processed.shape[0] > 1:
        cache_path = Path(LOCAL_DATA_PATH).joinpath(f"{DATA_PROCESSED_NAME}.csv")
        data_processed.to_csv(cache_path, header=True, index=False)
        print("✅ preprocessed data locally stored \n")

    upload_data_to_bq(
        data_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'{DATA_PROCESSED_NAME}',
        truncate=True
    )

    print("✅ preprocess() done \n")

def prepare_basic_clean_data(df: pd.DataFrame) -> pd.DataFrame:

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
    #                     Balancing Check                    #
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
    print('basic cleaning start ...')
    s = time.time()
    df['text'] = df['text'].apply(basic_cleaning)
    print(df.info())
    time_to_clean = time.time() - s
    print('Time for basic cleaning {:.2f} s'.format(time_to_clean))

    print("✅ data cleaned")

    return df

def preprocess_feature(text: str) -> str:

    # Tokenisation
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = _delete_stop_words(tokens)

    # Lemmatisation
    tokens = _lemmatize_text(tokens)

    # Combine words into a string
    processed_text = ' '.join(tokens)

    return processed_text

def basic_cleaning(text: str) -> str:
    # stripping:
    text = text.strip()

    # tolower:
    text = text.lower()

    # digit: Remove digits
    text = ''.join(char for char in text if not char.isdigit())

    # punctuation: Remove all punctuation marks
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)

    # delete html-tags
    text = re.sub('<[^<]+?>', '', text)

    return text


# Tokenise and remove stop words
def _delete_stop_words(X):
    stop_words = set(stopwords.words('english'))
    return [word for word in X if word.lower() not in stop_words and word not in string.punctuation]

# apply Lemmatization-Funktion
def _lemmatize_text(X):
    lemmatizer = WordNetLemmatizer()
    # Verben lemmatisieren
    verb_lemmatized = [lemmatizer.lemmatize(word, pos="v") for word in X]
    # Nomen lemmatisieren
    return [lemmatizer.lemmatize(word, pos="n") for word in verb_lemmatized]
