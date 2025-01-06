import glob
import os
import time
import pickle
import joblib
import tensorflow as tf
from fake_news_detection.params import *
from colorama import Fore, Style
from typing import TypeVar
from keras.models import Model
from keras.models import Sequential

# Define a generic type
T = TypeVar('T', Model, Sequential)

def save_model(model_type: str , model: T):
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", model_type, f"{timestamp}.h5")
    joblib.dump(model, model_path)

    print(f"✅ Model {model_type} saved locally")

def load_model(model_type: str) -> T:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    Return None (but do not Raise) if no model is found
    """
    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", model_type)
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model {model_type} from disk..." + Style.RESET_ALL)

    if model_type == 'baseline':
        return joblib.load(most_recent_model_path_on_disk)

    return tf.keras.models.load_model(most_recent_model_path_on_disk)

def save_results(model_type: str, params: dict, metrics: dict):
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", model_type, timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)
        print("✅ Params saved locally")

    # Save metrics locally
    if  metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", model_type, timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)
        print("✅ Results saved locally")

def load_metrics(model_type: str):
    """
    Return a saved metrics:
    - locally (latest one in alphabetical order)
    Return None (but do not Raise) if no model is found

    """
    print(Fore.BLUE + f"\nLoad latest {model_type} metrics from local registry..." + Style.RESET_ALL)

    # Get the latest base_score version name by the timestamp on disk
    local_metrics_directory = os.path.join(LOCAL_REGISTRY_PATH, "metrics", model_type)
    local_metrics_paths = glob.glob(f"{local_metrics_directory}/*")

    if not local_metrics_paths:
        return None

    most_recent_metrics_path_on_disk = sorted(local_metrics_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest {model_type} metrics from disk..." + Style.RESET_ALL)

    with open(most_recent_metrics_path_on_disk, "rb") as file:
        metrics = pickle.load(file)

    print(f"✅ {model_type} metrics loaded from local disk")

    return metrics
