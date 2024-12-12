import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from params import *
from fake_news_detection.ml_logic.data import get_data_with_cache, load_data_to_bq
from fake_news_detection.ml_logic.model import initialize_base_model, train_basic_model
from fake_news_detection.ml_logic.preprocessor import prepare_basic_clean_data, preprocess_features
from fake_news_detection.ml_logic.registry import save_base_model, save_model_results, load_base_model

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
    df = df.head(20)

    # cleaning data
    df = prepare_basic_clean_data(df)

    X = df.drop('label', axis=1)
    y = df['label']

    X_processed = preprocess_features(X['text'])

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()
    X_processed = X_processed.to_numpy().reshape(-1, 1)  # Form wird zu (100, 1)
    y = y.to_numpy().reshape(-1, 1)  # Form wird ebenfalls zu (100, 1)

    data_processed = pd.DataFrame(np.concatenate((X_processed, y), axis=1), columns=COLUMN_NAMES_RAW)

    # Store as CSV localy if at least one valid line is processed
    if data_processed.shape[0] > 1:
        cache_path = Path(LOCAL_DATA_PATH).joinpath(f"{DATA_PROCESSED_NAME}.csv")
        df.to_csv(cache_path, header=True, index=False)
        print("✅ preprocessed data locally stored \n")

    load_data_to_bq(
        data_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'{DATA_PROCESSED_NAME}',
        truncate=True
    )

    print("✅ preprocess() done \n")

def train_base_model(
        split_ratio: float = 0.02 # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return score accurancy as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # Below, our columns are called [text, label] on BQ
    # Query PROCESSED data from BigQuery using `get_data_with_cache`
    query = f"""SELECT {",".join(COLUMN_NAMES_RAW)} FROM {GCP_PROJECT}.{BQ_DATASET}.{DATA_PROCESSED_NAME}"""

    # Retrieve data using `get_data_with_cache`
    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"{DATA_PROCESSED_NAME}.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    train_length = int(len(data_processed)*(1-split_ratio))

    data_processed_train = data_processed.iloc[:train_length, :].sample(frac=1).to_numpy()
    data_processed_val = data_processed.iloc[train_length:, :].sample(frac=1).to_numpy()

    X_train_processed = data_processed_train[:, :-1]
    y_train = data_processed_train[:, -1]

    X_val_processed = data_processed_val[:, :-1]
    y_val = data_processed_val[:, -1]

    # Train model using `model.py`
    model_NB = load_base_model()

    if model_NB is None:
        model_NB = initialize_base_model(input_shape=X_train_processed.shape[1:])

    model_NB = train_basic_model(model_NB, X_train_processed, y_train)

    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed),
    )

    #save_model_results(params=params, metrics=dict(mae=val_mae))

    # by base bas model we don't habe a history to store
    save_base_model(model=model_NB)

    print("✅ train() base model done \n")

def pred_base_model(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))

    model_NB = load_base_model()
    assert model_NB is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model_NB.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred

preprocess()
