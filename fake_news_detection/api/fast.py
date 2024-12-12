from fastapi import FastAPI

app = FastAPI()

#@app.get("/")
#def root():
    #return {
   # 'greeting': 'Hello'
#}


@app.get("/")
async def root():
    return {"message": "API is live"}

@app.post("/predict")
async def predict(text):
    model=None
    #model.predict(process_data)
    #Add prediction logic here
    return {"prediction": "fake"}  # Example response
