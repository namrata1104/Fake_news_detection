from fake_news_detection.params import *
from fake_news_detection.ml_logic.preprocessor import preprocess
from fake_news_detection.ml_logic.data import get_data_with_cache, upload_data_to_bq
from fake_news_detection.ml_logic.models.model_factory import model_factory
from fake_news_detection.ml_logic.registry import load_metrics

#preprocess()
#model_factory.getModel(BASELINE).train(split_ratio=0.03)
model_factory.getModel(BASELINE).predict('Germany, officially the Federal Republic of Germany, is a country in Central Europe. It lies between the Baltic and North Sea to the north and the Alps to the south.')

#model_factory.getModel(LSTM).train(split_ratio=0.03)
model_factory.getModel(LSTM).predict('Germany, officially the Federal Republic of Germany, is a country in Central Europe. It lies between the Baltic and North Sea to the north and the Alps to the south.')

#model_factory.getModel(RNN).train(split_ratio=0.03)
model_factory.getModel(RNN).predict('Germany, officially the Federal Republic of Germany, is a country in Central Europe. It lies between the Baltic and North Sea to the north and the Alps to the south.')
