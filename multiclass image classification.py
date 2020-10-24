import pathlib
import PIL
from PIL import Image
import tensorflow as tf
import numpy as np
from numpy import asarray
import keras as k
from keras.models import Model


def create_input_output_matrix():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                       fname='flower_photos',
                                       untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    roses = list(data_dir.glob('roses/*'))
    daisy = list(data_dir.glob('daisy/*'))
    dandelion = list(data_dir.glob('dandelion/*'))
    sunflowers = list(data_dir.glob('sunflowers/*'))
    tulips = list(data_dir.glob('tulips/*'))
    X = np.empty((image_count, 50, 50, 3))
    Y = np.empty((image_count, 5))

    for i in range(0, len(roses)):
        image = Image.open(str(roses[i]))
        image1_resized = image.resize((50, 50))
        data = asarray(image1_resized)
        X[i, :, :, :] = data
        Y[i, :] = [1, 0, 0, 0, 0]

    for i in range(0, len(dandelion)):
        image = Image.open(str(dandelion[i]))
        image1_resized = image.resize((50, 50))
        data = asarray(image1_resized)
        X[i, :, :, :] = data
        Y[i, :] = [0, 1, 0, 0, 0]

    for i in range(0, len(daisy)):
        image = Image.open(str(daisy[i]))
        image1_resized = image.resize((50, 50))
        data = asarray(image1_resized)
        X[i, :, :, :] = data
        Y[i, :] = [0, 0, 1, 0, 0]

    for i in range(0, len(sunflowers)):
        image = Image.open(str(sunflowers[i]))
        image1_resized = image.resize((50, 50))
        data = asarray(image1_resized)
        X[i, :, :, :] = data
        Y[i, :] = [0, 0, 0, 1, 0]

    for i in range(0, len(tulips)):
        image = Image.open(str(tulips[i]))
        image1_resized = image.resize((50, 50))
        data = asarray(image1_resized)
        X[i, :, :, :] = data
        Y[i, :] = [0, 0, 0, 0, 1]


    np.savez("multi_class_image_training_data", X=X, Y=Y)


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
    y = k.layers.Dense(5, activation="softmax")(x)
    model = Model(inputs=[x_input], outputs=[y])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X, Y, epochs=8, batch_size=256)
    if save:
        model.save("basic_model")
    test(model)
    return model


def test(model):
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                       fname='flower_photos',
                                       untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    roses = list(data_dir.glob('roses/*'))
    daisy = list(data_dir.glob('daisy/*'))
    dandelion = list(data_dir.glob('dandelion/*'))
    sunflowers = list(data_dir.glob('sunflowers/*'))
    tulips = list(data_dir.glob('tulips/*'))
    pred = np.empty((5, 50, 50, 3))

    image = Image.open(str(roses[0]))
    image_resized = image.resize((50, 50))
    data = asarray(image_resized)
    pred[0, :, :, :] = data

    image = Image.open(str(daisy[0]))
    image_resized = image.resize((50, 50))
    data = asarray(image_resized)
    pred[1, :, :, :] = data

    image = Image.open(str(dandelion[0]))
    image_resized = image.resize((50, 50))
    data = asarray(image_resized)
    pred[2, :, :, :] = data

    image = Image.open(str(sunflowers[0]))
    image_resized = image.resize((50, 50))
    data = asarray(image_resized)
    pred[3, :, :, :] = data

    image = Image.open(str(tulips[0]))
    image_resized = image.resize((50, 50))
    data = asarray(image_resized)
    pred[4, :, :, :] = data
    print(pred)
    print(model.predict(pred))


create_input_output_matrix()
create_neural_network("multi_class_image_training_data.npz", save=False)