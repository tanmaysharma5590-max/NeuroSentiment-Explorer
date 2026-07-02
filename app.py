import os
import warnings
import pickle
import numpy as np
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

warnings.filterwarnings("ignore", message=".*TensorFlow GPU support is not available.*")
warnings.filterwarnings("ignore", message=".*Compiled the loaded model, but the compiled metrics.*")
warnings.filterwarnings("ignore", message=".*tf.reset_default_graph.*")

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Silence TensorFlow warning logs
if hasattr(tf, "get_logger"):
    tf.get_logger().setLevel("ERROR")

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

# ======================================================
# Load Resources
# ======================================================

@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5", compile=False)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()

# ======================================================
# Prediction Function
# ======================================================

def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    prediction = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(prediction)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "No prediction"

# ======================================================
# Main Title
# ======================================================

st.title("🧠 Next Word Prediction")

st.markdown(
    """
Predict the **next word** using a trained **LSTM Neural Network**.

Developed by **Tanmay Sharma**
"""
)

st.markdown("---")

# ======================================================
# User Input
# ======================================================

user_input = st.text_input(
    "✍️ Enter a sentence",
    placeholder="Example: Machine learning is"
)

# ======================================================
# Prediction Button
# ======================================================

if st.button("🔮 Predict Next Word"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        with st.spinner("Predicting..."):

            next_word = predict_next_word(user_input)

        st.success(f"### Predicted Next Word: **{next_word}**")

# ======================================================
# Footer
# ======================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center;color:gray;'>

### Next Word Prediction using LSTM

Developed by <b>Tanmay Sharma</b>

B.Tech | Machine Learning Project

</div>
""",
    unsafe_allow_html=True,
)