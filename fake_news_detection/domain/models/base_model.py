from fake_news_detection.domain.models.abstract_model import abstract_model
from keras.models import Model


class base_model(abstract_model[Model]):

    def __init__(self, model_type: str, model: Model):
        super().__init__(model_type, model)
        self.model_type = model_type
        self.model = model

"""
import numpy as np
from colorama import Fore, Style

    def score(self,
            X_test_processed: np.ndarray,
            y_test: np.ndarray):
        print(Fore.BLUE + f"🚀\n start scoring {self.model_type} model start..." + Style.RESET_ALL)
        score = self.model.score(X_test_processed.astype(str).flatten().tolist(), np.array(y_test).astype(bool).flatten())
        print(Fore.BLUE + f"\n end scoring {self.model_type}" + Style.RESET_ALL)
        return score """
