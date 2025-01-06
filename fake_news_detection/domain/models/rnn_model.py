from keras.models import Sequential
from fake_news_detection.domain.models.abstract_model import abstract_model

class rnn_model(abstract_model[Sequential]):

    def __init__(self, model_type: str, model: Sequential):
        super().__init__(model_type, model)
        self.model_type = model_type
        self.model = model

"""
import tensorflow as tf

    def compile(self):
        # Adjust the Adam optimizer parameters
        adam_optimizer = tf.keras.layers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self.model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
"""
