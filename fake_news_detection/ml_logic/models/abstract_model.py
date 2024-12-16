from typing import TypeVar, Union, Generic
from keras.models import Sequential
from keras.models import Model
import numpy as np

# Definiere einen Typ, der entweder 'Model' oder 'Sequential' sein kann
M = TypeVar('M', bound=Union[Model, Sequential])

class abstract_model(Generic[M]):

    def __init__(self, model: M):
        self.model = model

    def train(self, split_ratio: float = 0.03) -> float:
        pass

    def score(self,
            X_test_processed: np.ndarray,
            y_test: np.ndarray):
        pass

    def predict(self, news: str) -> tuple:
        pass

    def evaluate(
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64):
        pass

    def compile(learning_rate=0.0005):
        pass
