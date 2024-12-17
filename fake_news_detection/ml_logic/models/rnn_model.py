import numpy as np
from keras.models import Sequential
from fake_news_detection.ml_logic.models.abstract_model import abstract_model
import tensorflow as tf
from colorama import Fore, Style
from fake_news_detection.params import *
from fake_news_detection.ml_logic.registry import load_metrics
from fake_news_detection.ml_logic.preprocessor import basic_cleaning, preprocess_feature

class rnn_model(abstract_model[Sequential]):
    def __init__(self, model_type: str, model: Sequential):
        super().__init__(model_type, model)
        self.model_type = model_type
        self.model = model

    def compile(self):
        # Adjust the Adam optimizer parameters
        adam_optimizer = optimizer=tf.keras.layers.Adam()(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self.model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
