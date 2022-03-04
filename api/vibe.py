from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# import joblib
import PIL as Image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"Message": "They see me rollin'... They hatin'"}

@app.get("/predict")
def predict(picture):

    im = Image.open(picture).convert('L')
    im = im.resize((48, 48), Image.ANTIALIAS)
    im = np.array(im)

    return {'Pixels':im}
