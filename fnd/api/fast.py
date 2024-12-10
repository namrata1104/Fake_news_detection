import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from taxifare.ml_logic.registry import load_model
from taxifare.ml_logic.preprocessor import preprocess_features


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(
        title: str,  # 'news title'
        text: str   # 'news text'
    ):
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    """
    Make a prediction using the latest trained model
    """
    print("\n⭐️ Use case: predict")


    #pickup_datetime_utc = pd.Timestamp(datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")).tz_localize("US/Eastern")
    #pickup_datetime_iso = pickup_datetime_utc.isoformat()  # Für Serialisierung geeignet
    #pickup_datetime = pd.Timestamp(datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")).tz_localize("US/Eastern")
    X_pred = pd.DataFrame({
        'pickup_datetime':pd.Timestamp(pickup_datetime, tz='US/Eastern'),
        'pickup_longitude':[pickup_longitude],
        'pickup_latitude':[pickup_latitude],
        'dropoff_longitude':[dropoff_longitude],
        'dropoff_latitude':[dropoff_latitude],
        'passenger_count':[passenger_count]})

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return {
        'fare': float(y_pred)
    }




@app.get("/")
def root():
    return {
    'greeting': 'Hello fake news detector user !'
}
