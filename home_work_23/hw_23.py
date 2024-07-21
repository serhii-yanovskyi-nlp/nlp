import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential

max_features = 20000
max_len = 500

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
model = Sequential([
    Embedding(max_features, 128, input_length=max_len),
    Conv1D(64, 7, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(64, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Точність на тестових даних: {test_acc:.4f}')
print(f'Помилка на тестових даних: {test_loss:.4f}')