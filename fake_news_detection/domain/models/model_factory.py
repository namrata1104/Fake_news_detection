from fake_news_detection.domain.models.base_model import base_model
from fake_news_detection.domain.models.rnn_model  import rnn_model
from fake_news_detection.domain.models.lstm_model import lstm_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from fake_news_detection.domain.models.registry import load_model
from keras.models import Sequential
from sklearn.pipeline import Pipeline
from fake_news_detection.params import *
import tensorflow as tf

class model_factory:
    # Statische Instanzen für Modelle
    _base_model: base_model = None
    _rnn_model: rnn_model = None
    _lstm_model: lstm_model = None

    @staticmethod
    def initialize_models():
        """
        Initializes the models for use.
        This method is automatically called before the factory is utilized.
        """
        print("🚀 start: initialize all Models...")

        # BASELINE
        if model_factory._base_model is None:
            print("🚀 start: loading baseline model from disk...")
            model_factory._base_model = base_model(BASELINE, load_model(BASELINE))
            print("✅ end: loading baseline model from disk...")

        if model_factory._base_model is None or model_factory._base_model.model is None:
            print("🚀 start: creating new baseline model...")
            model = model_factory.create_base_model()    # model the text
            model_factory._base_model = base_model(BASELINE, model)
            print("✅ end: creating new baseline model...")

        # RNN
        if model_factory._rnn_model is None:
            print("🚀 start: loading rnn model from disk...")
            model_factory._rnn_model = rnn_model(RNN, load_model(RNN))
            print("✅ end: loading rnn model from disk...")
        if model_factory._rnn_model is None or model_factory._rnn_model.model is None:
            print("🚀 start: creating new rnn model ...")
            model_factory._rnn_model = rnn_model(RNN, model_factory.create_rnn_model())
            print("✅ end: creating new rnn model...")

        # LSTM
        if model_factory._lstm_model is None:
            print("🚀 start: loading lstm model from disk...")
            model_factory._lstm_model = lstm_model(LSTM, load_model(LSTM))
            print("✅ end: loading lstm model from disk...")
        if model_factory._lstm_model is None or model_factory._lstm_model.model is None:
            print("🚀 start: creating new lstm model...")
            model_factory._lstm_model = lstm_model(LSTM, model_factory.create_lstm_model())
            print("✅ end: creating new lstm model...")

        print("✅ end: initialize all Models...")

    @staticmethod
    def getModel(model_type: str):
        """
        Returns an instance of the specified model.

        Args:
            model_type (str): Type of the model to be created ('baseline', 'rnn', 'smtp').

        Returns:
            Model: An instance of the corresponding model.
        """
        if model_type == BASELINE:
            return model_factory._base_model
        elif model_type == RNN:
            return model_factory._rnn_model
        elif model_type == LSTM:
            return model_factory._lstm_model
        else:
            raise ValueError(f"Undefined Modelltyp: {model_type}")

    def create_base_model():
        return Pipeline([
                        ("tfidf",TfidfVectorizer()), # convert words to numbers using tfidf
                        ("clf",MultinomialNB())])

    def create_rnn_model():
        # Instantiate the model
        rnn_model = Sequential()

        # Add an embedding layer
        rnn_model.add(tf.keras.layers.Embedding(input_dim=5000, output_dim=128))

        # Add a simple RNN layer with 128 units
        rnn_model.add(tf.keras.layers.SimpleRNN(units=128, return_sequences=False))

        # Add dropout layers to prevent overfitting
        rnn_model.add(tf.keras.layers.Dropout(rate=0.5))

        # Add another dense layer and dropout layer
        rnn_model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        rnn_model.add(tf.keras.layers.Dropout(rate=0.5))

        # Add the final dense output layer with sigmoid activation
        rnn_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        return rnn_model

    def create_lstm_model():
        # Instantiate the LSTM model
        lstm_model = Sequential()

        # Add an embedding layer
        lstm_model.add(tf.keras.layers.Embedding(input_dim=5000, output_dim=128, input_length=500))

        # Add the LSTM layer
        lstm_model.add(tf.keras.layers.LSTM(units=128, return_sequences=False))

        # Add a dropout layer to prevent overfitting
        lstm_model.add(tf.keras.layers.Dropout(rate=0.2))

        # Add the dense output layer
        lstm_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        return lstm_model

# Automatic initialization of models when importing the class
model_factory.initialize_models()
