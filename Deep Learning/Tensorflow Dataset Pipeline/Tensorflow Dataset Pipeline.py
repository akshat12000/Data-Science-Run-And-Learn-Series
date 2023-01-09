import tensorflow as tf
import os

# Create a dataset from files
dataset = tf.data.Dataset.list_files("./reviews/*/*.txt",shuffle=False)

for files in dataset:
    print(files)

def extract_info(file_path):
    # Extract the label from the file path
    label = tf.strings.split(file_path, os.path.sep)[-2]
    # Read the file
    text = tf.io.read_file(file_path)
    return text, label

# Map the function to the dataset
dataset = dataset.map(extract_info)


# Filtering empty reviews
def filter_empty(text, label):
    return text!=""

dataset = dataset.filter(filter_empty)

# Shuffle the dataset
dataset = dataset.shuffle(2)

for text, label in dataset:
    print("Review: ", text.numpy()[:50])
    print("Label: ", label.numpy())

