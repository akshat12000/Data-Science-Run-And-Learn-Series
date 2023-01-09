import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf
from tensorflow import keras
import pathlib
from sklearn.model_selection import train_test_split

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# data_dir = keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
# data_dir = os.path.join(os.path.dirname(data_dir), 'flower_photos')

data_dir = pathlib.Path('.\\datasets\\flower_photos')

image_count = len(list(data_dir.glob('*/*.jpg')))

# roses = PIL.Image.open(str(list(data_dir.glob('roses/*'))))
# tulips = PIL.Image.open(str(list(data_dir.glob('tulips/*'))))

flower_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'tulips': list(data_dir.glob('tulips/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*'))
}

flower_labels_dict = {
    'roses': 0,
    'tulips': 1,
    'daisy': 2,
    'dandelion': 3,
    'sunflowers': 4
}

X,Y = [],[]

for flower_name, images in flower_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (180,180))
        X.append(resized_img)
        Y.append(flower_labels_dict[flower_name])

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

X_train_scaled = X_train/255
X_test_scaled = X_test/255

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(180,180,3)),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomContrast(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1)
])

num_classes = 5

model = keras.Sequential([
    data_augmentation,
    keras.layers.Conv2D(16, (3,3), padding="same", activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32, (3,3), padding="same", activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), padding="same", activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, Y_train, epochs=10)

print(model.evaluate(X_test_scaled, Y_test))















