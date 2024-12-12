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
    """
    Initialize the Neural Network with random weights
    """
    reg = regularizers.l1_l2(l2=0.005)

    model = Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(100, activation="relu", kernel_regularizer=reg))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.BatchNormalization(momentum=0.9))  # use momentum=0 to only use statistic of the last seen minibatch in inference mode ("short memory"). Use 1 to average statistics of all seen batch during training histories.
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(1, activation="linear"))

    print("✅ Model initialized")

    return model

def initialize_base_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create tokenization and modelling pipeline
    # MultinomialNB: Naive Bayes is a probabilistic classification model based on Bayes' theorem.
    # It is particularly effective for text classification tasks
    model_NB = Pipeline([
                        ("tfidf",TfidfVectorizer()), # convert words to numbers using tfidf
                        ("clf",MultinomialNB())])    # model the text

    print("✅ Model initialized")

    return model_NB

def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history

def train_basic_model(
        model_NB: Model,
        X_train_processed: np.ndarray,
        y_train: np.ndarray
    ) -> Model:
    """
    Fit the model and return fitted_model, history
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

    return model_NB


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
