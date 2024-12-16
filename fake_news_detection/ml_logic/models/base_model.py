from fake_news_detection.ml_logic.models.abstract_model import abstract_model
from keras.models import Model
import numpy as np
import pandas as pd
from colorama import Fore, Style
from fake_news_detection.params import *
from fake_news_detection.ml_logic.data import get_processed_data, split_data
from fake_news_detection.ml_logic.registry import save_model, save_results, load_metrics
from fake_news_detection.ml_logic.preprocessor import basic_cleaning, preprocess_feature

class base_model(abstract_model[Model]):

    def __init__(self, model: Model):
        super().__init__(model)
        self.model = model


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
        print(Fore.BLUE + f"\nTraining {BASELINE} model..." + Style.RESET_ALL)
        X_train_processed = X_train_processed.astype(str).flatten().tolist()
        print("✅ X_train_processed converted and validated:")

        # Validierung und Konvertierung von y_train
        y_train = np.array(y_train).astype(bool).flatten()
        print("✅ y_train converted and validated:", y_train[:2])  # Zeigt die ersten 10 Labels

        print(type(self.model))
        self.model.fit(X_train_processed, y_train)

        # save trained model
        save_model(BASELINE, self.model)

        # score model and store scoring results accuracy
        accuracy_score = self.score(X_test_processed, y_test)
        params = dict(context="train", training_set_size=DATA_SIZE, row_count=len(X_train_processed))
        save_results(model_type=BASELINE, params=params, metrics=dict(accuracy=accuracy_score))

        print(f"✅ train() {BASELINE} model done \n")

        return self

    def score(self,
            X_test_processed: np.ndarray,
            y_test: np.ndarray):
        print(Fore.BLUE + f"🚀\n start scoring {BASELINE} model start..." + Style.RESET_ALL)
        score = self.model.score(X_test_processed.astype(str).flatten().tolist(), np.array(y_test).astype(bool).flatten())
        print(Fore.BLUE + f"\n end scoring {BASELINE}" + Style.RESET_ALL)
        return score

    def predict(self, text: str) -> tuple:
        """
        Make a prediction using the latest trained model and return the prediction and accuracy.
        """
        print("\n⭐️ Use case: predict")

        # Preprocess the input text
        processed_text = preprocess_feature(basic_cleaning(text))

        # Convert the processed text into a DataFrame for the model
        news = pd.DataFrame([processed_text], columns=['text'])

        # Get the prediction
        print(Fore.BLUE + f"🚀\n start predicting {BASELINE} model start..." + Style.RESET_ALL)
        y_pred = self.model.predict(news)
        print(Fore.BLUE + f"✅\n end predicting {BASELINE}" + Style.RESET_ALL)

        # load accuracy from the last stored metrics
        accuracy = load_metrics(BASELINE).get("accuracy", None)

        print("\n✅ predict done: ", y_pred, y_pred.shape, " | Accuracy: ", accuracy, "\n")
        print(y_pred[0], accuracy)
        return (y_pred[0], accuracy)
