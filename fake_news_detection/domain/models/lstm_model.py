from keras.models import Sequential
from fake_news_detection.domain.models.abstract_model import abstract_model
from fake_news_detection.params import *

class lstm_model(abstract_model[Sequential]):

    def __init__(self, model_type: str, model: Sequential):
        super().__init__(model_type, model)
        self.model_type = model_type
        self.model = model

"""
import tensorflow as tf

    def compile(self):
        # Compile the model with Adam optimizer and binary cross-entropy loss
        self.model.compile(optimizer=tf.keras.layers.Adam(), loss='binary_crossentropy', metrics=['accuracy']) """
