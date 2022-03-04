import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#config
img = Image.open('haut_de_page.jpeg')
st.set_page_config(page_title = 'Get The Vibe', page_icon = img)

#title of the app
logo = Image.open('get the vibe .png')
st.image(logo)

#file downloader
file= st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

#shows the picture if there is one
if file == None :
    st.write('No image')
else:
    file_details = {"filename":file.name, "filetype":file.type,
                              "filesize (in MB)":file.size / 1000000}
    st.write(file_details)
    image = Image.open(file)
    st.image(image, width = 400)

#uploading image from your camera
st.header('or upload directly from your camera :')
picture = st.camera_input("Take a picture")
if picture:
     st.image(picture)

#hiding 'made with streamlit'
hide_ad = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_ad, unsafe_allow_html = True)



#if we want to download the file in the computer
#if choice == "Image":
#
#		st.subheader("Image")
#			type=["png","jpg","jpeg"])
#
#		if file is not None:
        # TO See details
#			  file_details = {"filename":file.name, "filetype":file.type,
 #                             "filesize":file.size}
#			  st.write(file_details)
#			  st.image(load_image(image_file), width=250)
#
#			  #Saving upload
#			  with open(os.path.join("fileDir",file.name),"wb") as f:
#			  	f.write((file).getbuffer())
#
#			  st.success("File Saved")
