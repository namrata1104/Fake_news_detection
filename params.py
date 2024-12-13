import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
DATA_RAW_NAME = os.environ.get("DATA_RAW_NAME")
DATA_PROCESSED_NAME = os.environ.get("DATA_PROCESSED_NAME")
#CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION  = os.environ.get("BQ_REGION")
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "namrata1104", "Fake_news_detection", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "namrata1104", "Fake_news_detection", "training_outputs")

COLUMN_NAMES_RAW = ['text','label']

DTYPES_RAW = {
    "title": "string",
    "text": "string",
    "label": "bool",
}

DTYPES_PROCESSED = {
    "text": "string",
    "label": "bool",
}
################## VALIDATIONS #################
