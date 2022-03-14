from email import header
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
from tensorflow import expand_dims
from retinaface import RetinaFace

from fastapi.encoders import jsonable_encoder
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import cv2 # Problem???
import joblib
import numpy as np
import pandas as pd
import json


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


@app.post("/vibecheck")

def image_filter(file: bytes = File(...)):

    emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear',
                   3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    image = PIL.Image.open(BytesIO(file))
    extension =  image.format

    numpy_image = np.array(image)
    img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    resp = RetinaFace.detect_faces(img,
                               allow_upscaling = False)

    faces_area = []
    faces = []

    for inner_dict in resp.values():
        face_rect = inner_dict.get('facial_area')
        faces_area.append(face_rect)
        facial_img = img[face_rect[1]: face_rect[3], face_rect[0]: face_rect[2]]
        faces.append(facial_img[:, :, ::-1])

    data = np.zeros(shape=(len(faces), 48, 48), dtype = 'float64')

    for i, face in enumerate(faces):
        face_bw = np.array(PIL.Image.fromarray(faces[i])
                           .convert("L")
                           .resize((48, 48), PIL.Image.ANTIALIAS))
        data[i]=face_bw

    data = (data/255) -0.5
    data = expand_dims(data, axis=-1)
    model = joblib.load('model.joblib')
    results = model.predict(data)

    emotion_results = np.vectorize(emotion_map.get)(results.argmax(axis = 1))
    emotion_values =  results.max(axis = 1)

    emotion_df = pd.DataFrame(data={'Emotion': emotion_map.values(),
                                    'Percentage': np.mean(results,axis=0)})

    emotion_df = (emotion_df
                  .sort_values(by='Percentage',ascending=False)
                  .head(3).to_dict())

    image_to_draw = PIL.ImageDraw.Draw(image)

    def emotion_colors(emotion):
        emotion_dict = {'Angry':'#ff0000','Digust':'#ff0000', 'Fear':'#ff0000',
                   'Happy':'#00ff00', 'Sad':'#ff0000',
                   'Surprise':'#ffff00','Neutral':'#ffff00'}
        return emotion_dict.get(emotion)



    for i,face_rect in enumerate(faces_area):
        color = emotion_colors(emotion_results[i])
        box_length = face_rect[2]-face_rect[0]
        text_size = int(np.ceil(box_length/7.5))

        outline_size = int(np.ceil(box_length/45))
        image_to_draw.rectangle(face_rect,
                                outline=color ,
                                width=outline_size)

        font_L = PIL.ImageFont.truetype("resources/OpenSans-SemiBold.ttf",
                                    text_size)
        text_bbox = f'{emotion_results[i]}: {emotion_values[i]:.3f}'

        image_to_draw.multiline_text((face_rect[0],face_rect[3]),font = font_L,
                                 text = text_bbox,  fill=color)

    filtered_image = BytesIO()
    image.save(filtered_image, extension)

    filtered_image.seek(0)

    return StreamingResponse(filtered_image,
                             headers={'emotion_df':str(emotion_df)},
                             media_type="image/jpeg")
