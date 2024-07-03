from fastapi import FastAPI
import pickle
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()

class InputAcceptor(BaseModel):
    date: str

@app.get('/')
def welcome():
    return "Welcome to Stock Price Prediction app"

@app.post("/StockPricePrediction/Predict")
def predict(input_data: InputAcceptor):
    try:
        with open("./DataFrames/20.9101lstmforecast.pkl", "rb") as f:
            df = pickle.load(f)
        
        # Extract the date part from the input date string
        input_date_str = input_data.date.split('T')[0]  # Get 'YYYY-MM-DD' part
        
        # Convert the input date string to a datetime object
        input_date = datetime.strptime(input_date_str, '%Y-%m-%d')

        # Get the price for the input date from the DataFrame
        price = df.loc[input_date_str]

        pricing = price.values
        
        # Return the price as a JSON response
        if pricing:
            return float(pricing)
        else:
            return "Might be a public Holiday or weekend if in range or maybe out of range."
    except Exception as e:
        # Return the error message as a JSON response
        return {"error": str(e)}
