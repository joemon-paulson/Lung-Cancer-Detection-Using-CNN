import streamlit as st
import os
import keras
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image

st.title("Lung Cancer Classification Using CNN")
st.text("Upload a scan for Classification")

def chestScanPrediction(path,_model):
    model = keras.models.load_model(_model)
    # Loading Image
    img = image.load_img(path, target_size=(350,350))
    # Normalizing Image
    norm_img = image.img_to_array(img)/255
    # Converting Image to Numpy Array
    input_arr_img = np.array([norm_img])
    # Getting Predictions
    pred = np.argmax(model.predict(input_arr_img))
    # Printing Model Prediction
    if pred == 0:
        st.write("The scan is adenocarcinoma")
    elif pred == 1:
        st.write("The scan is large.cell.carcinoma")
    elif pred==2:
        st.write("The scan is normal")
    else:
        st.write("This scan is squamous.cell.carcinoma")
    

uploaded_file = st.file_uploader("Choose a scan ...", type="png")

if uploaded_file is not None:
  
    
    data = Image.open(uploaded_file)
    st.image(data, caption='Uploaded Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    chestScanPrediction(uploaded_file, 'ct_resnet_best_model.hdf5')
    

