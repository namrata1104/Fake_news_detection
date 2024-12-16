from fake_news_detection.params import *
from fake_news_detection.ml_logic.preprocessor import preprocess
from fake_news_detection.ml_logic.data import get_data_with_cache, upload_data_to_bq
from fake_news_detection.ml_logic.models.model_factory import model_factory
from fake_news_detection.ml_logic.registry import load_metrics

#preprocess()
#model_factory.getModel(BASELINE).train(split_ratio=0.03)
#model_factory.getModel(BASELINE).predict('i am not a fake news')
