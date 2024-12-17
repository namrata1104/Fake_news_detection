import numpy as np
import tensorflow as tf
import pandas as pd

from keras.models import Sequential
from fake_news_detection.ml_logic.models.abstract_model import abstract_model
from colorama import Fore, Style
from fake_news_detection.params import *
from fake_news_detection.ml_logic.preprocessor import basic_cleaning, preprocess_feature
from fake_news_detection.ml_logic.data import get_processed_data, split_data
from fake_news_detection.ml_logic.registry import save_model, save_results

class lstm_model(abstract_model[Sequential]):

    def __init__(self, model_type: str, model: Sequential):
        super().__init__(model_type, model)
        self.model_type = model_type
        self.model = model

    def compile(self):
        # Compile the model with Adam optimizer and binary cross-entropy loss
        self.model.compile(optimizer=tf.keras.layers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
