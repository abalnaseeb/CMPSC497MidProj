import time
import psutil
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import streamlit as st


max_length = 200  # Maximum length of the input sequences
num_classes = 4  # AG News has 4 classes
class_labels=["World", "Sports", "Business", "Sci/Tech"]

process = psutil.Process()

# Load the model
loaded_model = load_model('../Base_Model/ag_news_cnn_glove_model.h5')

# Load the tokenizer
with open('../Base_Model/tokenizer.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)

st.title("News Classification App")
user_input = st.text_area("Enter news text:")

if st.button("Predict"):

    # Measure memory before inference
    memory_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
    start_time = time.time()
    sequence = loaded_tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = loaded_model.predict(padded_sequence)
    predicted_class = class_labels[np.argmax(prediction)]
    end_time = time.time()
    
    #Measure memory after inference
    memory_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    memory_used = memory_after - memory_before
    
      
    inference_time = (end_time - start_time) * 1000 # Convert to milliseconds
    
    st.write(f"Predicted Category: {predicted_class}")
    st.write(f"Inference Time: **{inference_time:.2f} ms**")
    st.write(f"Memory Usage: **{memory_used:.2f} MB**")
