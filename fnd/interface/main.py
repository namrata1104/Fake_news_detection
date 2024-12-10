import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from fnd.params import *
from fnd.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from fnd.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from fnd.ml_logic.preprocessor import preprocess_features
from fnd.ml_logic.registry import load_model, save_model, save_results
from fnd.ml_logic.registry import mlflow_run, mlflow_transition_model

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
    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WAGON}`.{BQ_DATASET}.raw_{DATA_SIZE}
        """

    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("data_raw", f"{DATA_FILENAME}.csv")
    data = get_data_with_cache(
        query=query,
        gcp_project= {GCP_PROJECT},
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # cleaning data
    data = clean_data(data)

    X = data['text']
    y = data['label']
    X_processed = preprocess_features(X)

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()
    data_processed = pd.DataFrame(np.concatenate((X_processed, y), axis=1))

    load_data_to_bq(
        data_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_{DATA_SIZE}',
        truncate=True
    )

    print("✅ preprocess() done \n")

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
        pickup_longitude=[-73.950655],
        pickup_latitude=[40.783282],
        dropoff_longitude=[-73.984365],
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


preprocess()
