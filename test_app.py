import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#title of the app
st.title("Get The Vibe")

#file downloader
file= st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

#shows the picture if there is one
if file == None :
    st.write('No image')
else:
    image = Image.open(file)
    st.image(image)

#uploading image from your camera
st.header('or upload directly from your camera :')
picture = st.camera_input("Take a picture")
if picture:
     st.image(picture)
