import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("spam.csv")

df["spam"] = df["Category"].apply(lambda x: 1 if x == "spam" else 0)

X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["spam"], test_size=0.2, random_state=42,stratify=df["spam"])

# Preprocess the data
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"

bert_preprocess = hub.KerasLayer(preprocess_url)
bert_encoder = hub.KerasLayer(encoder_url)

def get_sentence_embeddings(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)["pooled_output"]

print(get_sentence_embeddings(["500$ discount. hurry up","Bhavin, are you up for a volleyball game tomorrow?"]))

# Build the model
input_text = tf.keras.layers.Input(shape=(), dtype=tf.string,name="text")
preprocessed_text = bert_preprocess(input_text)
outputs = bert_encoder(preprocessed_text)["pooled_output"]
outputs = tf.keras.layers.Dropout(0.1,name="dropout")(outputs)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(outputs)
model = tf.keras.Model(inputs=input_text, outputs=outputs)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
print(model.evaluate(X_test, y_test))

# Predict the output
print(model.predict(["500$ discount. hurry up","Bhavin, are you up for a volleyball game tomorrow?"]))





