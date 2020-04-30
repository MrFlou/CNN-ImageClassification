# Neural net packages
import tensorflow as tf
from tensorflow.keras import models, layers, backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import os
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
#from PIL import ImageGrab
import cv2
import time
import argparse

batch_size = 32
epochs = 10
image_dementions = (100,100,3)


def build(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    # if we are using "channels first", update the input shape
    # and channels dimension
    if backend.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), padding="same",
        input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

#model = models.Sequential()
#model.add(layers.Conv2D(IMG_WIDTH, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(IMG_WIDTH*2, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(IMG_WIDTH*2, (3, 3), activation='relu'))
#model.add(layers.Flatten())
#model.add(layers.Dense(IMG_WIDTH*2, activation='relu'))


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
args = vars(ap.parse_args())


def gather_pokeindex(_imagePaths):
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(_imagePaths)))
    for imagePath in imagePaths:
    	# load the image, pre-process it, and store it in the data list
    	image = cv2.imread(imagePath)
    	image = cv2.resize(image, (image_dementions[1], image_dementions[0]))
    	image = img_to_array(image)
    	data.append(image)

    	# extract the class label from the image path and update the
    	# labels list
    	label = imagePath.split(os.path.sep)[-2]
    	labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    (test_dataset, validation_dataset, testY, validationY) = train_test_split(data,
    	labels, test_size=0.2, random_state=42)


    #test_dataset = image_generator.flow_from_directory(batch_size=batch_size,                                                           directory=os.path.join(imagePaths),                                      shuffle=True,                                                           color_mode="rgb",                             target_size=(image_dementions[0], image_dementions[1]),                                                           class_mode='categorical',                                                           subset='training')

    #validation_dataset = image_generator.flow_from_directory(batch_size=batch_size,                                                           directory=os.path.join(imagePaths),                                                           target_size=(image_dementions[0], image_dementions[1]),                                                           color_mode="rgb",                                                           class_mode='categorical',                                                           subset='validation')

    model = build(width=image_dementions[0], height=image_dementions[1],depth=image_dementions[2], classes=5)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
    #model.add(layers.Dense(len(test_dataset.class_indices)))

    #model = build(width=IMG_HEIGHT, height=IMG_HEIGHT,depth=3, classes=len(test_dataset.class_indices))
    #model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()
    return test_dataset,testY, validation_dataset, validationY, model

def train_Pokedex(_test_dataset, _validation_dataset, _model):
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    	horizontal_flip=True, fill_mode="nearest")
    history = _model.fit(
    aug.flow(_test_dataset,testY, batch_size=batch_size),
    steps_per_epoch=np.ceil(len(_test_dataset)/batch_size),
    epochs=epochs,
    validation_data=aug.flow(_validation_dataset,validationY),
    validation_steps=len(_validation_dataset) // batch_size
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    return history



test_dataset, testY, validation_dataset, validationY, model = gather_pokeindex(args["dataset"])
train_Pokedex(test_dataset, validation_dataset, model)



#def catch_Pokemon(pokemon_image):
    # Code to scan image and porocess #


#def train_Pokedex():
