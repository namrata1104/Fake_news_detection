from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fake_news_detection.ml_logic.models.model_factory import model_factory
from fastapi.encoders import jsonable_encoder
import numpy as np

app = FastAPI()

# Pydantic model for the request body
class TextRequest(BaseModel):
    text: str
    models: List[str]  # List of selected models (e.g. ["mstl", "base", "rnn"])

@app.get("/")
async def root():
    return {"message": "API is live"}

# Utility function to convert numpy types to native Python types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Converts numpy arrays to lists
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Converts numpy.bool_ to native Python bool
    elif isinstance(obj, np.generic):
        return obj.item()  # Converts other numpy types to native Python types
    return obj

@app.post("/predict")
async def predict(request: TextRequest):
    """
    Receives a text and a list of selected models,
    performs predictions, and returns the results directly as JSON.
    """
    # Dictionary to hold the results
    response = {}

    for model_key in request.models:
        # Get the prediction and accuracy
        y_pred, probability = model_factory.getModel(model_key).predict(request.text)

        # Convert to native Python types
        y_pred = convert_numpy_types(y_pred)
        probability = convert_numpy_types(probability)

        # In case probability is an array, use its first element
        if isinstance(probability, (list, np.ndarray)):
            probability = probability[0]  # Adjust if needed for multi-class

        # Define label based on y_pred
        label = "Real News" if y_pred == 1 else "Fake News"

        # Add the result to the response dictionary
        response[model_key] = {
            "label": label,  # Human-readable label
            "prediction": int(y_pred),  # 0 or 1
            "probability": probability if probability is not None else 0.0  # Fallback to 0.0 if probability is None
        }

    result = jsonable_encoder(response)
    print(f"result: {result}")
    # Return the response as JSON-serializable data
    return result
