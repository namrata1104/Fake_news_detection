import numpy as np
import pandas as pd
import time
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from fake_news_detection.params import *
from fake_news_detection.ml_logic.model import initialize_base_model, get_score_base_model, train_basic_model
from fake_news_detection.ml_logic.preprocessor import prepare_basic_clean_data, basic_cleaning, preprocess_feature
from fake_news_detection.ml_logic.registry import save_base_model, load_base_model, save_model_results, load_base_metrics
from fake_news_detection.ml_logic.data import get_data_with_cache, upload_data_to_bq

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

    print(X_processed.iloc[0])
    print(y.iloc[0])

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

def train_base_model(
        split_ratio: float = 0.03  # 0.03 represents ~ 1 month of validation data on a 2009-2015 train set
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return score accuracy as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed training data..." + Style.RESET_ALL)

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

    # Shuffle the dataset
    data_processed = data_processed.sample(frac=1).to_numpy()

    train_length = int(len(data_processed) * (1 - split_ratio))
    data_processed_train = data_processed[:train_length]
    data_processed_test = data_processed[train_length:]

    # Separate features (X_train_processed) and labels (y_train)
    X_train_processed = data_processed_train[:, :-1]
    y_train = data_processed_train[:, -1]

    # Separate test features (X_test) and labels (y_test)
    X_test_processed = data_processed_test[:, :-1]
    y_test = data_processed_test[:, -1]

    # Train model using `model.py`
    model_NB = load_base_model()

    if model_NB is None:
        model_NB = initialize_base_model(input_shape=X_train_processed.shape[1:])

    model_NB = train_basic_model(model_NB, X_train_processed, y_train)
    # Save the model results
    save_base_model(model = model_NB)

    # Calculate the accuracy on the test data set
    accuracy_score = get_score_base_model(model_NB, X_test_processed, y_test)
    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed))

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_model_results(params=params, metrics=dict(accuracy=accuracy_score))

    print("✅ train() base model done \n")

    return model_NB

def pred_base_model(text: str) -> tuple:
    """
    Make a prediction using the latest trained model and return the prediction and accuracy.
    """
    print("\n⭐️ Use case: predict")

    model_NB = load_base_model()
    assert model_NB is not None

    # Preprocess the input text
    processed_text = preprocess_feature(basic_cleaning(text))

    # Convert the processed text into a DataFrame for the model
    frame_one_text = pd.DataFrame([processed_text], columns=['text'])

    # Get the prediction
    y_pred = model_NB.predict(frame_one_text)
    print(y_pred)
    # load accuracy from the last stored metrics
    accuracy = load_base_metrics("local").get("accuracy", None)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, " | Accuracy: ", accuracy, "\n")

    return (y_pred[0], accuracy)


#preprocess()
#train_base_model()
pred_base_model('i am not a fake news')
