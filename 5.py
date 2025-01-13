import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample text
text = "Hello world! How are you doing today? The world is beautiful."

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

# Prepare input-output pairs
X, y = [], []
for i in range(1, len(sequences)):
    X.append(sequences[:i])
    y.append(sequences[i])

X = pad_sequences(X, padding='pre')
y = np.array(y)

# Build the LSTM model
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 10),
    LSTM(50),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=200, verbose=0)

# Predict next word
def predict_next_word(seed_text, n_words=1):
    for _ in range(n_words):
        token_list = pad_sequences([tokenizer.texts_to_sequences([seed_text])[0]], maxlen=X.shape[1], padding='pre')
        predicted = model.predict(token_list, verbose=0).argmax()
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text

# Test prediction
print(predict_next_word("Hello world", 3))
