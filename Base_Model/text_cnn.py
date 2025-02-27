import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
import time
import psutil

# Load the dataset from CSV files
train_df = pd.read_csv('../Data/train.csv', header=0, names=['class', 'title', 'description'])
test_df = pd.read_csv('../Data/test.csv', header=0, names=['class', 'title', 'description'])

# Combine title and description for better context
train_df['text'] = train_df['title'] + " " + train_df['description']
test_df['text'] = test_df['title'] + " " + test_df['description']

# Extract texts and labels
train_texts = train_df['text'].tolist()
train_labels = train_df['class'].to_numpy() - 1  # Convert classes to 0-based index

test_texts = test_df['text'].tolist()
test_labels = test_df['class'].to_numpy() - 1  # Convert classes to 0-based index

# Parameters
vocab_size = 70338  # Size of the vocabulary
embedding_dim = 100  # Dimension of the word embeddings
max_length = 200  # Maximum length of the input sequences
num_classes = 4  # AG News has 4 classes

# Tokenize the text data
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_texts)

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences to ensure uniform input size
X_train = pad_sequences(train_sequences, maxlen=max_length)
X_test = pad_sequences(test_sequences, maxlen=max_length)

# Convert labels to numpy arrays
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# Load GloVe embeddings
embeddings_index = {}
with open('../Data/glove.6B/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build the CNN model with GloVe embeddings
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length,
              weights=[embedding_matrix], trainable=False),  # Use pre-trained GloVe embeddings
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
])

# Build the model explicitly
model.build(input_shape=(None, max_length))

# Summary of the model
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

process = psutil.Process() # to analyse memeory usage

memory_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB
start_time = time.time()

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

end_time = time.time()
memory_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB

memory_used = memory_after - memory_before
train_time = end_time - start_time

print("\n")
print("tran time : ",train_time,"Seconds")
print("memory_used : ", memory_used,"MB")
print("\n")


# Plot training and validation accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


# Classification report
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["World", "Sports", "Business", "Sci/Tech"]))


confusion_mat =  metrics.confusion_matrix(y_test, y_pred)
individual_class_acc   =  confusion_mat.diagonal()/confusion_mat.sum(axis = 1)
print("\nIndividual class Accuracy : \n", individual_class_acc )
        
plt.figure(figsize=(15, 8))
plt.bar(['World', 'Sports', 'Business', 'Sci/Tech'], individual_class_acc)
plt.xlabel('News')
plt.ylabel('Accuracy')
plt.title('Class Wise Accuracy');
plt.show()


disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_mat,display_labels= ['World', 'Sports', 'Business', 'Sci/Tech'])
disp.plot()
plt.title('Confusion Matrix');
plt.show() 

# Make predictions with the loaded model
new_texts = ["Apple launches new iPhone", "The stock market reached an all-time high today.","The president announced a new policy to address climate change.","Football team wins the championship.","Scientists discovered a new planet in a distant galaxy.","The team won the championship after a thrilling match."]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_X = pad_sequences(new_sequences, maxlen=max_length)
predictions = model.predict(new_X)
class_labels = ['World', 'Sports', 'Business', 'Sci/Tech']
predicted_classes = [class_labels[np.argmax(pred)] for pred in predictions]
print("Predictions:", predicted_classes)


# Save the model
model.save('ag_news_cnn_glove_model.h5')
print("Model saved!")

# Save the tokenizer (for reuse during inference)
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved!")

