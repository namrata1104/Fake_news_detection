from keras.models import Sequential
import numpy as np
from fake_news_detection.ml_logic.models.abstract_model import abstract_model

class rnn_model(abstract_model[Sequential]):
    def __init__(self, model: Sequential):
        super().__init__(model)

    def train(self,
        X_train_processed: np.ndarray,
        y_train: np.ndarray):
        pass

    def score(self,
            X_test_processed: np.ndarray,
            y_test: np.ndarray):
        pass

    pass
    def predict(self, news: str):
        return (
            False,
             0.95
            )
