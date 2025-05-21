import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
   
    # This function sets the background of a Streamlit app to an image specified by the given image file.

    # Parameters:
    #     image_file (str): The path to the image file to be used as the background.

    # Returns:
    #     None
    
    with open(image_file, "rb") as f:
        img_data = f.read()
    
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
    <style>
      .stApp {{
        background-image: url(data:image/png;base64,{b64_encoded});
        background-size: cover;
      }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """

    # convert image to (299, 299)
    image = ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS)

    # 2) convert to uint8   <-- NO manual scaling
    img_array = np.asarray(image, dtype=np.uint8)

    # 3) add batch dimension
    data = np.expand_dims(img_array, axis=0)      # shape (1, 200, 200, 3)

    # 4) predict
    preds = model.predict(data)
    idx   = np.argmax(preds[0])
    return class_names[idx], float(preds[0][idx])