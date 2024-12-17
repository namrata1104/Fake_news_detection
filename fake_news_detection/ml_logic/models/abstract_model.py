from typing import TypeVar, Union, Generic
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
import numpy as np
import pandas as pd

from colorama import Fore, Style
from fake_news_detection.ml_logic.preprocessor import basic_cleaning, preprocess_feature
from fake_news_detection.ml_logic.registry import save_model, save_results
from fake_news_detection.ml_logic.data import get_processed_data, split_data
from sklearn.metrics import accuracy_score
from fake_news_detection.params import *
# Definiere einen Typ, der entweder 'Model' oder 'Sequential' sein kann
M = TypeVar('M', bound=Union[Model, Sequential])

class abstract_model(Generic[M]):

    def __init__(self, model_type: str, model: M):
        self.model = model
        self.model_type = model_type

    def predict(self, news: str) -> tuple:
        """
        Make a prediction using the latest trained model and return the prediction and accuracy.
        """
        print("\n⭐️ Use case: predict")

        # Preprocess the input text
        processed_news = preprocess_feature(basic_cleaning(news))

        if self.model_type == BASELINE:
            # Convert the processed text into a DataFrame for the model
             X_pred = pd.DataFrame([processed_news], columns=['text'])
        elif self.model_type == RNN or self.model_type == LSTM:
            # Tokenize and pad sequences
            tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(processed_news)
            X_seq = tokenizer.texts_to_sequences(processed_news)
            # Pad sequences to ensure uniform input length
            X_pred = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=500)

        # Get the prediction
        print(Fore.BLUE + f"🚀\n start predicting {self.model_type} model start..." + Style.RESET_ALL)
        y_pred = self.model.predict(X_pred)
        #y_pred = (y_pred > 0.5).astype(int)

        # Wahrscheinlichkeitswert für die positive Klasse (z.B. "Fake" oder "True")
        print(y_pred)
        if self.model_type == BASELINE:
            probability = y_pred[0]
        elif self.model_type == RNN or self.model_type == LSTM:
            probability = y_pred[0][0]

        # Ausgabe der Vorhersagewahrscheinlichkeit
        print("\n✅ predict done: ", y_pred, y_pred.shape, " | probability: ", probability, "\n")
        print(Fore.BLUE + f"✅ end predicting {self.model_type}" + Style.RESET_ALL)

        return (y_pred[0], probability)


    def train(self, split_ratio: float = 0.03) -> float:
        """
        - Download processed data from your BQ table (or from cache if it exists)
        - Train on the preprocessed dataset (which should be ordered by date)
        - Store training results and model weights
        Return score accuracy as a float
        """
        print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
        print(Fore.BLUE + "\nLoading preprocessed training data..." + Style.RESET_ALL)

        data_processed = get_processed_data()

        if data_processed.shape[0] < 10:
            print("❌ Not enough processed data retrieved to train on")
            return None

        # Shuffle the dataset
        data_processed = data_processed.sample(frac=1).to_numpy()

        X_train_processed, y_train, X_test_processed, y_test = split_data(data_processed, split_ratio)

        #  Fit the model and return fitted_model
        print(Fore.BLUE + f"\nTraining {self.model_type} model..." + Style.RESET_ALL)
        X_train_processed = X_train_processed.astype(str).flatten().tolist()
        print("✅ X_train_processed converted and validated:")

        # Validierung und Konvertierung von y_train
        y_train = np.array(y_train).astype(bool).flatten()
        print("✅ y_train converted and validated:", y_train[:2])  # Zeigt die ersten 10 Labels

        print(type(self.model))

        if self.model_type == BASELINE:
            self.model.fit(X_train_processed, y_train)
            params = dict(context="train", training_set_size=DATA_SIZE, row_count=len(X_train_processed))
            accuracy = self.score(X_test_processed, y_test)

        elif self.model_type == LSTM:
            self.model.fit(X_train_processed, y_train, epochs=5, batch_size=64, validation_data=(X_test_processed, y_test))
            params = dict(context="train", training_set_size=DATA_SIZE, row_count=len(X_train_processed))
            _, accuracy = self.model.evaluate(X_test_processed, y_test)

        elif self.model_type == RNN:
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            # Train the model with early stopping and model checkpoint
            history_1 = self.model.fit(
                X_train_processed, y_train,
                epochs=10,
                batch_size=64,
                validation_data=(X_test_processed, y_test),
                callbacks=[early_stopping]
            )
            # Evaluate the model on the test data
            params = dict(context="train", training_set_size=DATA_SIZE, row_count=len(X_train_processed))
            _, accuracy = self.model.evaluate(X_test_processed, y_test)

        # save trained model
        save_model(self.model_type, self.model)
        # save result
        save_results(model_type=self.model_type, params=params, metrics=dict(accuracy=accuracy))

        print(f"✅ train() {self.model_type} model done \n")

        return self
