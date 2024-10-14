from flask import Flask
from typing import Optional, List
import pandas as pd
from pydantic import BaseModel, conlist
import pickle
from Backend.model import output
from model import dataset, recommand, extracted_data

dataset=pd.read_csv('../DataSet/dataset.csv',compression='gzip')

app = Flask(__name__)

class params(BaseModel):
    n_neighbors:int=5
    return_distance:bool=False

class PredictionIn(BaseModel):
    nutrition_input:conlist(float)
    ingredients:list[str]=[]
    params:Optional[params]


class Recipe(BaseModel):
    Name:str
    CookTime:str
    PrepTime:str
    TotalTime:str
    RecipeIngredientParts:list[str]
    Calories:float
    FatContent:float
    SaturatedFatContent:float
    CholesterolContent:float
    SodiumContent:float
    CarbohydrateContent:float
    FiberContent:float
    SugarContent:float
    ProteinContent:float
    RecipeInstructions:list[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None

@app.get("/")
async def root():
    return {"health_check": "OK"}

@app.post("/predict/",response_model=PredictionOut)
def update_item(prediction_input:PredictionIn):
    recommendation_dataframe=recommand(dataset,prediction_input.nutrition_input,prediction_input.ingredients,prediction_input.params.dict())
    output=recommendation_dataframe
    if output is None:
        return {"output":None}
    else:
        return {"output":output}

pickle.dump(open('diet.pkl'),'wb')

if __name__ == '__main__':
   app.run()