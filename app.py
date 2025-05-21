# --- at the very top of main.py ---
import streamlit as st
import tensorflow as tf
from PIL import Image
from util import classify, set_background
import os
import pathlib

MODEL_DIR   = "model"
MODEL_FILE  = "runtime_model.keras"
LABELS_FILE = "labels.txt"
MODEL_ID    = "1bGlCzrxRBIYmqPHwM1JHioZVc8V25uzr"           # <-- your Drive file-ID
LABELS_ID   = "189ASX31O13FX6U0N1HgaTmkmWJat1JG2"            # <-- labels file-ID

def gdrive_download(file_id, out_path):
    try:
        import gdown
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out_path, quiet=False)

# fetch once per container
pathlib.Path(MODEL_DIR).mkdir(exist_ok=True)
if not os.path.exists(f"{MODEL_DIR}/{MODEL_FILE}"):
    st.info("Downloading model…")
    gdrive_download(MODEL_ID, f"{MODEL_DIR}/{MODEL_FILE}")
if not os.path.exists(f"{MODEL_DIR}/{LABELS_FILE}"):
    gdrive_download(LABELS_ID, f"{MODEL_DIR}/{LABELS_FILE}")

set_background('./bg.jpg')

# set title
st.title('Crack Detection')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

model_path  = os.path.join(MODEL_DIR, MODEL_FILE)    # CHANGED: use os.path.join instead of hard-coded string
labels_path = os.path.join(MODEL_DIR, LABELS_FILE)


@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(model_path)    # CHANGED: now points to model_path
    with open(labels_path, 'r') as f:                 # CHANGED: now points to labels_path
        # CHANGED: simplified parsing and removed explicit f.close()
        names = [line.strip().split(maxsplit=1)[1] for line in f]
    return model, names

# single call to load both
model, class_names = load_model_and_labels()

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}".format(conf_score))
