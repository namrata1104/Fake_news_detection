from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fake_news_detection.ml_logic.models.model_factory import model_factory

app = FastAPI()

# Pydantic model for the request body
class TextRequest(BaseModel):
    text: str
    models: List[str]  # List of selected models (e.g. ["mstl", "base", "rnn"])

@app.get("/")
async def root():
    return {"message": "API is live"}

@app.post("/predict")
async def predict(request: TextRequest):
    """
    Receives a text and a list of selected models,
    performs predictions, and returns the results directly as JSON.
    """
    # Dictionary to hold the results
    response = {}
    for model_key in request.models:
        # Assuming model_factory.getModel(model_key) returns a model instance that has a 'predict' method
        model = model_factory.getModel(model_key)

        # Get the prediction and accuracy
        y_pred, accuracy = model.predict(request.text)
        print('y_pred: ' + f"{y_pred}")
        print('accuracy: ' + f"{accuracy}")

        # Add the result to the response dictionary
        response[model_key] = {
            "prediction": bool(y_pred),
            "accuracy": accuracy if accuracy is not None else 0  # default accuracy to 0 if None
        }

    return response
