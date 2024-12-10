import os
import numpy as np

##################  VARIABLES  ##################
DATA_FILENAME = 'WELFake_Dataset'
DATA_SIZE = os.environ.get("DATA_SIZE")
#CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
#MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
BQ_DATASET = os.environ.get("BQ_DATASET")
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".code", "namrata1104", "Fake_news_detection", "raw_data")

COLUMN_NAMES_RAW = ['text','label']

DTYPES_RAW = {
    "title": "object",
    "text": "object",
    "label": "int64",
}
################## VALIDATIONS #################
