import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predictions_arr = [round(100 * i, 2) for i in predictions[0]]
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, predictions_arr

model = tf.keras.models.load_model('potato_model.h5', compile=False)

def main():
    st.set_page_config(page_title="Potato Disease Classifier")
    st.sidebar.title("Potato Disease Classifier")
    st.sidebar.info("Upload an image of a potato leaf to detect early or late blight.")
    st.title("Potato Disease Detection")
    uploaded_file = st.file_uploader("Upload a potato leaf image",type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image,caption="Uploaded Image",use_column_width=True)
        image = image.resize((256,256))
        img_arr = np.array(image)
        predicted_class,predictions=predict(model,img_arr)

        response = {
            "predicted_class": predicted_class,
            "early": f"{predictions[0]:.2f}%",
            "late": f"{predictions[1]:.2f}%",
            "healthy": f"{predictions[2]:.2f}%"
        }


        st.success(f"Predicted Class : {response['predicted_class']}",icon="✅")
        st.write("Probabilities:")
        col1,col2,col3 = st.columns(3)
        col1.metric("Early Blight" , f"{response['early']}", f"{response['early']}")
        col2.metric("Late Blight" , f"{response['late']}", f"{response['late']}")
        col3.metric("Healthy" , f"{response['healthy']}", f"{response['healthy']}")

if __name__ == "__main__":
    main()
