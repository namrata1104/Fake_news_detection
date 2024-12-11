import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict
@app.get("/predict")
def predict(
        title: str,  # 'need help'
        text: str   # 'very urgent! i'm not a fake news'
    ):
    return {
        'label': 'true'
    }


@app.get("/")
def root():
    return {
    'greeting': 'Hello fake news detector user !'
}
