from fastapi import FastAPI
from pydantic import BaseModel
from fake_news_detection.interface.main import pred_base_model

app = FastAPI()

# Pydantic Modell für den Body der Anfrage
class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "API is live"}

@app.post("/predict")
async def predict(request: TextRequest):
    """
    Receive a text input, pass it to the model for making predict,
    and return the result and accuracy as response.
    """
    # Call pred_base_model to get the prediction and accuracy

    y_pred, accuracy = pred_base_model(request.text)

    # Create the response dictionary
    response = {
        "prediction": bool(y_pred),
        "accuracy": accuracy
    }

    return response
