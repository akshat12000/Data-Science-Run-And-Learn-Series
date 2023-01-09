import numpy as np 
import tensorflow as tf

reviews = ['nice food',
            'amazing restaurant',
            'too good',
            'just loved it!',
            'will go again',
            'horrible food',
            'never go there',
            'poor service',
            'poor quality',
            'needs improvement']

sentiment = np.array([1,1,1,1,1,0,0,0,0,0])
vocab_size = 30
encoded_reviews = [tf.keras.preprocessing.text.one_hot(d, vocab_size) for d in reviews]

max_length = 3
padded_reviews = tf.keras.preprocessing.sequence.pad_sequences(encoded_reviews, maxlen=max_length, padding='post')

embedded_vector_size = 4

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedded_vector_size, input_length=max_length, name='embedding'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
model.fit(padded_reviews, sentiment, epochs=50, verbose=0)

weights = model.get_layer('embedding').get_weights()[0]
print(weights)
