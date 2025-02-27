import sys
import time
import psutil
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import streamlit as st

# Model dimension, filename and title
models = {
    "50D": 
    {
        "filename": "ag_news_cnn_glove_model_50D.h5",
        "title": "News Classification App 50-D"
    },
    "100D": 
    {
        "filename": "ag_news_cnn_glove_model_100D.h5",
        "title": "News Classification App 100-D"
    },
    "200D": 
    {
        "filename": "ag_news_cnn_glove_model_200D.h5",
        "title": "News Classification App 200-D"
    },
    "300D": 
    {
        "filename": "ag_news_cnn_glove_model_300D.h5",
        "title": "News Classification App 300-D"
    }
}

# Exit the program if correct model dimension is not passed
if len(sys.argv) != 2 or sys.argv[1] not in models:
    print("Either model dimension is not specified or incorrectly specified")
    print("You can pass one of the following model dimensions")
    print("1. 50D")
    print("2. 100D")
    print("3. 200D")
    print("4. 300D")
    sys.exit(-1)

model = models[sys.argv[1]]
max_length = 200  # Maximum length of the input sequences
num_classes = 4  # AG News has 4 classes
class_labels=["World", "Sports", "Business", "Sci/Tech"]

process = psutil.Process()

# Load the model
loaded_model = load_model('../Differnt_GloVe_Dimension/' + model["filename"])

# Load the tokenizer
with open('../Differnt_GloVe_Dimension/tokenizer.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)

st.title(model["title"])
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
