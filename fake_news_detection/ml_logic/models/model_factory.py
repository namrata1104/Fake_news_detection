from fake_news_detection.ml_logic.models.base_model import base_model
from fake_news_detection.ml_logic.models.rnn_model  import rnn_model
from fake_news_detection.ml_logic.models.lstm_model import lstm_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from fake_news_detection.ml_logic.registry import load_model
from keras.models import Sequential
from sklearn.pipeline import Pipeline
from fake_news_detection.params import *

class model_factory:
    # Statische Instanzen für Modelle
    _base_model: base_model = None
    _rnn_model: rnn_model = None
    _lstm_model: lstm_model = None

    @staticmethod
    def initialize_models():
        """
        Initialisiert die Modelle für den Einsatz.
        Diese Methode wird automatisch aufgerufen, bevor die Factory verwendet wird.
        """
        print("🚀 start: initialize all Models...")

        # BASELINE
        if model_factory._base_model is None:
            print("🚀 start: loading baseline model from disk...")
            model_factory._base_model = base_model(load_model(BASELINE))
            print("✅ end: loading baseline model from disk...")

        if model_factory._base_model is None or model_factory._base_model.model is None:
            print("🚀 start: creating new baseline model...")
            model = Pipeline([
                        ("tfidf",TfidfVectorizer()), # convert words to numbers using tfidf
                        ("clf",MultinomialNB())])    # model the text
            model_factory._base_model = base_model(model)
            print("✅ end: creating new baseline model...")

        # RNN
        if model_factory._rnn_model is None:
            print("🚀 start: loading rnn model from disk...")
            model_factory._rnn_model = rnn_model(load_model(RNN))
            print("✅ end: loading rnn model from disk...")
        if model_factory._rnn_model is None or model_factory._rnn_model.model is None:
            print("🚀 start: creating new rnn model ...")
            model_factory._rnn_model = rnn_model(Sequential())
            print("✅ end: creating new rnn model...")

        # LSTM
        if model_factory._lstm_model is None:
            print("🚀 start: loading lstm model from disk...")
            model_factory._lstm_model = lstm_model(load_model(LSTM))
            print("✅ end: loading lstm model from disk...")
        if model_factory._lstm_model is None or model_factory._lstm_model.model is None:
            print("🚀 start: creating new lstm model...")
            model_factory._lstm_model = lstm_model(Sequential())
            print("✅ end: creating new lstm model...")

        print("✅ end: initialize all Models...")

    @staticmethod
    def getModel(model_type: str):
        """
        Gibt eine Instanz des angegebenen Modells zurück.

        Args:
            model_type (str): Typ des zu erstellenden Modells ('baseline', 'rnn', 'smtp').

        Returns:
            Model: Eine Instanz des entsprechenden Modells.
        """
        if model_type == BASELINE:
            return model_factory._base_model
        elif model_type == RNN:
            return model_factory._rnn_model
        elif model_type == LSTM:
            return model_factory._lstm_model
        else:
            raise ValueError(f"Unbekannter Modelltyp: {model_type}")

# Automatisches Initialisieren der Modelle beim Importieren der Klasse
model_factory.initialize_models()

# Beispiel: Verwendung der Factory
if __name__ == "__main__":
    # Erstelle ein Baseline-Modell
    print("Baseline-Modell erstellt:", model_factory.getModel(BASELINE))

    # Erstelle ein RNN-Modell
    print("RNN-Modell erstellt:", model_factory.getModel(RNN))

    # Erstelle ein LSTM-Modell
    print("LSTM-Modell erstellt:", model_factory.getModel(LSTM))
