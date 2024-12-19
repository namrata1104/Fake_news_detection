from fake_news_detection.params import *
from fake_news_detection.ml_logic.preprocessor import preprocess
from fake_news_detection.ml_logic.data import get_data_with_cache, upload_data_to_bq
from fake_news_detection.ml_logic.models.model_factory import model_factory
from fake_news_detection.ml_logic.registry import load_metrics

#text ="house intel chair on trumprussia fake story “no evidence of anything” video'"
#text='dr ben carson targeted by the irs “i never had an audit until i spoke at the national prayer breakfast”dr ben carson tells the story of what happened when he spoke out against obama'
#text='watch hilarious ad calls into question health of aging clinton crime family bossesafter watching this telling video you ll wonder if instead of working so hard to get back into the white house hillary s time would be better spent looking into an assisted living situation for her and bill'
#text='Germany is a country in europe'

#model_factory.getModel(BASELINE).train(split_ratio=0.03)
#model_factory.getModel(BASELINE).predict(text)

#model_factory.getModel(RNN).train(split_ratio=0.03)
#model_factory.getModel(RNN).predict(text)

#model_factory.getModel(LSTM).train(split_ratio=0.03)
#model_factory.getModel(LSTM).predict(text)
