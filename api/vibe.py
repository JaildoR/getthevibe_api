from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from tensorflow import expand_dims

import joblib
import PIL.Image
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

    im = PIL.Image.open(picture).convert('L')
    im = im.resize((48, 48), PIL.Image.ANTIALIAS)
    im = np.array(im)

    return {'Pixels':im}

@app.post("/")

def image_filter(file: bytes = File(...)):
    pil_image = (PIL.Image.open(BytesIO(file))
                 .convert('L')
                 .resize((48, 48), PIL.Image.ANTIALIAS))

    # filtered_image = BytesIO()
    # pil_image.save(filtered_image, "JPEG")
    # filtered_image.seek(0)

    # return StreamingResponse(filtered_image, media_type="image/jpeg")
    emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear',
                   3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    im = np.array(pil_image)
    im = (im/255) -0.5

    im = expand_dims(im, axis=0)
    im = expand_dims(im, axis=-1)


    model = joblib.load('model.joblib')

    results = model.predict(im)[0]


    return {'emotion':emotion_map.get(results.argmax())}
