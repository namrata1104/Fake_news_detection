import os
import numpy as np
/home/lahrech/code/namrata1104/Fake_news_detection/raw_data
##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".code", "namrata1104", "Fake_news_detection", "raw_data")

COLUMN_NAMES_RAW = ['text','label']

DTYPES_RAW = {
    "text": "string",
    "label": "bool"
}
################## VALIDATIONS #################
