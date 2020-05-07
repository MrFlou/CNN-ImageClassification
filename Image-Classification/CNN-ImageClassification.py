# Neural net packages
import tensorflow as tf
from tensorflow.keras import models, layers, backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle

import os
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from PIL import ImageGrab
import datetime
import random
import cv2
import argparse
import pydot
import graphviz

batch_size = 32
epochs = 64
image_dementions = (100,100,3)
log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)
print(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def VGGNet(width, height, depth, classes):
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

def CNNNet(width, height, depth, classes):
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
    model.add(Conv2D(32, (10, 10), padding="same",input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten(input_shape=inputShape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-s", "--state", required=True,
	help="Train or Scan")
args = vars(ap.parse_args())


def gather_pokeindex(_imagePaths):
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(_imagePaths)))
    random.seed(42)
    random.shuffle(imagePaths)
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)

        image = cv2.resize(image, (image_dementions[1], image_dementions[0]))
        image = img_to_array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    (test_dataset, validation_dataset, testY, validationY) = train_test_split(data,
    	labels, test_size=0.2, random_state=42)

    model = VGGNet(width=image_dementions[0], height=image_dementions[1],depth=image_dementions[2], classes=5)
    model.compile(optimizer=Adam(lr=0.001, decay=0.001 / epochs), loss="categorical_crossentropy", metrics=['accuracy'])
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.summary()

    with file_writer.as_default():
        img = np.reshape(data[0:25], (-1, 100, 100, 3))
        tf.summary.image("Training data", img, max_outputs=25, step=0)
    f = open(os.path.dirname(__file__) + "/pokedexlabels.pickle", "wb")
    f.write(pickle.dumps(lb))
    f.close()
    return test_dataset, testY, validation_dataset, validationY, model

def train_Pokedex(_test_dataset, _validation_dataset, _model):
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    	horizontal_flip=True, fill_mode="nearest")
    history = _model.fit(
    aug.flow(_test_dataset,testY, batch_size=batch_size),
    steps_per_epoch=len(_test_dataset)//batch_size,
    epochs=epochs,
    validation_data=(_validation_dataset,validationY),
    callbacks=[tensorboard_callback]
    )

    model.evaluate(_validation_dataset, validationY, verbose=2)
    model.save(os.path.join(os.path.dirname(__file__), "pokedexModel.model"))
    return history




if (args["state"] == "Train" or args["state"] == "train"):
    test_dataset, testY, validation_dataset, validationY, model = gather_pokeindex(args["dataset"])
    train_Pokedex(test_dataset, validation_dataset, model)
if (args["state"] == "scan" or args["state"] == "Scan"):
    dir = os.path.join(os.path.dirname(__file__), "pokedexModel.model")
    modelpredict = tf.keras.models.load_model(dir)
    modelpredict.compile(optimizer=Adam(lr=0.001, decay=0.001 / epochs), loss="categorical_crossentropy", metrics=['accuracy'])
    dir2 = os.path.join(os.path.dirname(__file__), "pokedexlabels.pickle")
    lb = pickle.loads(open(dir2, "rb").read())
    modelpredict.summary()
    while(True):
        printscreen =  np.array(ImageGrab.grab(bbox=(200,200,800,800)))
        image = cv2.resize(printscreen, (100, 100))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        prop = modelpredict.predict(image)[0]
        idx = np.argmax(prop)
        label = lb.classes_[idx]
        label2 = "{}: {:.2f}% ".format(label, prop[idx] * 100)
        print("[INFO] {}".format(label2))
        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
