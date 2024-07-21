import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K


def cosine_similarity(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.sum(x * y, axis=-1, keepdims=True)

def build_embedding_model(vocab_size, embedding_dim, lstm_units):
    input = Input(shape=(None,))
    x = Embedding(vocab_size, embedding_dim)(input)
    x = LSTM(lstm_units)(x)
    x = Dense(embedding_dim)(x)
    return Model(input, x)


# Построение модели Siamese
def build_siamese_model(vocab_size, embedding_dim, lstm_units):
    embedding_model = build_embedding_model(vocab_size, embedding_dim, lstm_units)

    input_a = Input(shape=(None,))
    input_b = Input(shape=(None,))

    embedded_a = embedding_model(input_a)
    embedded_b = embedding_model(input_b)

    similarity = Lambda(cosine_similarity)([embedded_a, embedded_b])

    model = Model(inputs=[input_a, input_b], outputs=similarity)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Пример данных
sentences = [
    "I love machine learning",
    "Machine learning is amazing",
    "I enjoy learning about deep learning",
    "Natural language processing is a subset of AI",
]

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')
sentences_a = padded_sequences[:2]  # Первые два предложения
sentences_b = padded_sequences[2:4]  # Следующие два предложения
labels = np.array([1, 0])  # 1 если предложения похожи, 0 если нет
vocab_size = 5000
embedding_dim = 50
lstm_units = 64
model = build_siamese_model(vocab_size, embedding_dim, lstm_units)
model.fit([sentences_a, sentences_b], labels, epochs=5)
test_sentences_a = np.array([padded_sequences[0]])  # Преобразованные последовательности индексов
test_sentences_b = np.array([padded_sequences[1]])
predictions = model.predict([test_sentences_a, test_sentences_b])

def is_similar(predictions, threshold=0.5):
    return predictions >= threshold
for pred in predictions:
    print(f"Similarity Score: {pred[0]:.2f}")

