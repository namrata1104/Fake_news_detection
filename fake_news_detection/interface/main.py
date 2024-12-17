from fake_news_detection.params import *
from fake_news_detection.ml_logic.preprocessor import preprocess
from fake_news_detection.ml_logic.data import get_data_with_cache, upload_data_to_bq
from fake_news_detection.ml_logic.models.model_factory import model_factory
from fake_news_detection.ml_logic.registry import load_metrics

#text = 'UNBELIEVABLE! OBAMAâ€™S ATTORNEY GENERAL SAYS MOST CHARLOTTE RIOTERS WERE â€œPEACEFULâ€ PROTESTERSâ€¦In Her Home State Of North Carolina [VIDEO] Now, most of the demonstrators gathered last night were exercising their constitutional and protected right to peaceful protest in order to raise issues and create change.    Loretta Lynch aka Eric Holder in a skirt'
#text = 'law enforcement high alert follow threat cop white blacklivesmatter fyf terrorist videono comment expect barack obama member fyf fukyoflag blacklivesmatter movement call lynch hang white people cop encourage others radio show tuesday night turn tide kill white people cop send message kill black peop'
#text =  "MSNBC, DNC reach deal to host Democratic debate in New Hampshire 'The tentative deal reached this weekend between the presidential campaigns of Hillary Clinton and Bernie Sanders includes a debate Thursday at the University of New Hampshire in Durham. MSNBC announced Sunday it will host the debate, scheduled for 9 p.m. Eastern with Chuck Todd and Rachel Maddow moderating. New Hampshire's first-in-the nation primary is Feb. 9. Clinton and Sanders are in a tight race before Monday's Iowa caucuses, and Clinton trails the Vermont senator in New Hampshire, raising the possibility that the Democratic front-runner could lose the first two contests. Former Maryland Gov. Martin O'Malley has trailed them by wide margins. The Democratic National Committee says it's reached an agreement in principal to have the party sanction and manage more debates during the primary schedule, including the New Hampshire debate."
#model_factory.getModel(BASELINE).train(split_ratio=0.03)
text = 'Germany is a country in europe'
model_factory.getModel(BASELINE).predict(text)

#model_factory.getModel(LSTM).train(split_ratio=0.03)
model_factory.getModel(LSTM).predict(text)

#model_factory.getModel(RNN).train(split_ratio=0.03)
model_factory.getModel(RNN).predict(text)
print(text)
#UNBELIEVABLE! OBAMAâ€™S ATTORNEY GENERAL SAYS MOST CHARLOTTE RIOTERS WERE â€œPEACEFULâ€%
