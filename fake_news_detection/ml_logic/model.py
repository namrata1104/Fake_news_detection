import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from keras import Model, Sequential, layers, regularizers, optimizers


end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model(input_shape: tuple) -> Model:

    # TODO: please implement me with (RNN or SMTP) model
    model = Sequential()
    print("✅ Model not initialized")
    return model

def compile_model(model: Model, learning_rate=0.0005) -> Model:
    # TODO: please implement me with (RNN or SMTP) model
    print("✅ Model not compiled")

    return model

def train_model():
    # TODO: please implement me with (RNN or SMTP) model
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(f"✅ Model not trained yet !")

    return None

def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:

    # TODO: please implement me with (RNN or SMTP) model
    """
    Evaluate trained model performance on the dataset
    """
    print(f"✅ Model not evaluated yet !")
    return None

def initialize_base_model(input_shape: tuple) -> Model:
    """
    Initialize the base model
    """
    # Create tokenization and modelling pipeline
    # MultinomialNB: Naive Bayes is a probabilistic classification model based on Bayes' theorem.
    # It is particularly effective for text classification tasks
    model_NB = Pipeline([
                        ("tfidf",TfidfVectorizer()), # convert words to numbers using tfidf
                        ("clf",MultinomialNB())])    # model the text

    print("✅ Base Model initialized")

    return model_NB

def train_basic_model(
        model_NB: Model,
        X_train_processed: np.ndarray,
        y_train: np.ndarray
    ) -> Model:
    """
    Fit the model and return fitted_model
    """
    print(Fore.BLUE + "\nTraining base model..." + Style.RESET_ALL)

    history = model_NB.fit(X_train_processed, y_train)

    # Access to Modell and the Vectorizer after Fit
    # 1. Access to Naive Bayes Modell (MultinomialNB)
    clf = model_NB.named_steps["clf"]
    print("✅Logarithmised class probabilities (class_log_prior_):")
    print(clf.class_log_prior_) # Shows the log probabilities of each class
    print("✅\nLogarithmised feature probabilities (feature_log_prob_):")
    print(clf.feature_log_prob_) # Shows the probabilities per feature (word)

    # 2nd access to the TfidfVectoriser
    vectoriser = model_NB.named_steps["tfidf"]
    print("✅\nFeatures (words) in the vocabulary of the TfidfVectoriser:")
    print(vectoriser.get_feature_names_out())  # Lists all words in the vocabulary
    print(Fore.BLUE + "\n✅ base model trained !" + Style.RESET_ALL)
    return model_NB
