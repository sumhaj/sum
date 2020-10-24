import PIL
from PIL import Image
import numpy as np
from numpy import asarray
import keras as k
from keras.models import Model
import glob


def create_input_output_matrix():
    X = np.empty((40, 50, 50, 3))
    Y = np.empty((40, 1))
    i = 0
    output = 1

    for images in glob.glob('./training images/pen images/*'):
        image1 = Image.open(images)
        image1_resized = image1.resize((50, 50))
        data = asarray(image1_resized)
        X[i, :, :, :] = data
        Y[i, 0] = output
        i += 1

    for images in glob.glob('./training images/without pen images/*'):
        output = 0
        image1 = Image.open(images)
        image1_resized = image1.resize((50, 50))
        data = asarray(image1_resized)
        X[i, :, :, :] = data
        Y[i, 0] = output
        i += 1
    np.savez("image_training_data", X=X, Y=Y)


def create_neural_network(training_data, save=False):
    data = np.load(training_data)
    X = data['X'].copy()
    Y = data['Y'].copy()

    x_input = k.Input((50, 50, 3))
    x = k.layers.Conv2D(filters=12, kernel_size=5, padding="same", data_format="channels_last")(x_input)
    x = k.layers.AveragePooling2D(pool_size=(3, 3), data_format="channels_last")(x)
    x = k.layers.Conv2D(filters=12, kernel_size=5, padding="same")(x)
    x = k.layers.AveragePooling2D(pool_size=(3, 3), data_format="channels_last")(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(250)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Dropout(rate=0.3)(x)
    x = k.layers.Dense(100)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Dropout(rate=0.3)(x)
    x = k.layers.Dense(25)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Dropout(rate=0.3)(x)
    y = k.layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[x_input], outputs=y)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X, Y, epochs=8)
    if save:
        model.save("basic_model")
    test(model)
    return model


def test(model):
    pred = np.empty((4, 50, 50, 3))
    i=0

    for images in glob.glob('./test images/*'):
        print(images)
        image1 = Image.open(images)
        image1_resized = image1.resize((50, 50))
        data = asarray(image1_resized)
        pred[i, :, :, :] = data
        i += 1

    print(model.predict(pred))


create_input_output_matrix()
create_neural_network("image_training_data.npz", save=False)