import streamlit as st
import numpy as np
from keras.models import load_model
import cv2
from PIL import Image

st.set_page_config(layout = 'centered')

st.title("# Document Classifier")
st.subheader("About the Model")
st.write("""The NN is designed to classify different types of documents
            which can be emails, resume, scientific publications. Kera's Generators are used to do perform
            loading and preprocessing on train and test datasets. It's a simple model which uses a combination of Convo2D layers
            , MaxPool layers , furthermore 2 dropout layers to improve accuracy and prevent overfitting.""")

st.subheader("How to Use")
st.write("1.Upload an image")
st.write("2.The image should be clear")
st.write("3.Lastly click on the classify button to classify")


model = load_model("model.h5")

image = st.file_uploader('Upload a Image file', type = ['jpg','png','jpeg'])
if image is None:
    st.text('Upload an Image file')

else:
    img= Image.open(image)
    
    if st.button("Predict"):
    # fetching the uploaded image and preprocessing it
        img = Image.open(image)
        img = img.resize((256,256))
        st.write(np.array(img).shape)
        img = np.reshape(256,256,3)
        st.image(img)
        
        
        pred = model.predict(img)     
        st.write(np.argmax(pred))
        # predict the digit
        # st.write((model.predict(img_arr)))