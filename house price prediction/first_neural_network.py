import csv
import numpy

import keras as k
from keras.models import Model


def Create_input_output_matrix():
    filename = "house_data.csv"

    headers = []
    list1 = []

    file1 = "train_set_norm"

    with open(filename, 'r') as file:
        read = csv.reader(file)
        headers = next(read)

        for r in read:
            list1.append(r)

    # input of training data set
    X = numpy.empty([40000, 4, 1])

    # output of training data set
    Y = numpy.empty([40000, 1])

    for i in range(0, len(list1)):
        condition_map = {'very bad': -2, 'bad': -1, 'neutral': 0, 'good': 1, 'very good': 2}
        area_rating_map = {'*': 1, '**': 2, '***': 3, '****': 4, '*****': 5}
        size = int(list1[i][0])
        bedrooms = int(list1[i][1])
        area_rating = area_rating_map[list1[i][2]]
        condition = condition_map[list1[i][3]]
        cost = float(list1[i][4])
        X[i, :, 0] = [float(size / 1000), bedrooms, area_rating, condition]
        Y[i, :] = [float(cost / 1000)]

    numpy.savez(file1, X=X, Y=Y)

    # def Create_neural_network(training_data,save = False):
    #     data = numpy.load(training_data)
    #     X = data['X'].copy()
    #     Y = data['Y'].copy()
    #     x_input = k.Input((4, 1))
    #     x = k.layers.Flatten()(x_input)
    #     x = k.layers.Dense(3)(x)
    #     x = k.layers.Activation('relu')(x)
    #     x = k.layers.Dropout(rate=0.3)(x)
    #     y = k.layers.Dense(1, activation='relu')(x)
    #
    #     model = Model(inputs=x_input, outputs=y)
    #     model.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['accuracy'])
    #     #     model.summary()
    #     model.fit(X, Y, epochs=16)
    #     if save:
    #         model.save("basic_model")
    #     test(model)
    #     return model

    # def Create_neural_network1(training_data, save = False):
    #         data = numpy.load(training_data)
    #         X = data['X'].copy()
    #         Y = data['Y'].copy()
    #         x_input = k.Input((4, 1))
    #         x = k.layers.Flatten()(x_input)
    #         x = k.layers.Dense(10)(x)
    #         x = k.layers.Activation('relu')(x)
    #         x = k.layers.Dropout(rate=0.3)(x)
    #         x = k.layers.Dense(5, activation='relu')(x)
    #         x = k.layers.Dropout(rate=0.2)(x)
    #         y = k.layers.Dense(1, activation='relu')(x)
    #
    #         model = Model(inputs=x_input, outputs=y)
    #         model.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['accuracy'])
    #         model.summary()
    #         model.fit(X, Y, epochs=16, batch_size=64)
    #         if save:
    #             model.save("descending_model")
    #         test(model)
    #         return model

    # def Create_neural_network2(training_data, save = False):
    #         data = numpy.load(training_data)
    #         X = data['X'].copy()
    #         Y = data['Y'].copy()
    #         x_input = k.Input((4, 1))
    #         x = k.layers.Flatten()(x_input)
    #         x = k.layers.Dense(10)(x)
    #         x = k.layers.Activation('relu')(x)
    #         x = k.layers.Dropout(rate=0.3)(x)
    #         x = k.layers.Dense(20, activation='relu')(x)
    #         x = k.layers.Dropout(rate=0.25)(x)
    #         y = k.layers.Dense(1, activation='relu')(x)
    #
    #         model = Model(inputs=x_input, outputs=y)
    #         model.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['accuracy'])
    #         model.summary()
    #         model.fit(X, Y, epochs=16, batch_size=64)
    #         if save:
    #             model.save("complex_model")
    #         test(model)
    #         return model

    # def Create_neural_network3(training_data, save=False):
    #     data = numpy.load(training_data)
    #     X = data['X'].copy()
    #     Y = data['Y'].copy()
    #     x_input = k.Input((4, 1))
    #     x = k.layers.Flatten()(x_input)
    #     x = k.layers.Dense(20)(x)
    #     x = k.layers.Activation('relu')(x)
    #     x = k.layers.Dropout(rate=0.25)(x)
    #     x = k.layers.Dense(10, activation='relu')(x)
    #     x = k.layers.Dropout(rate=0.3)(x)
    #     y = k.layers.Dense(1, activation='relu')(x)
    #
    #     model = Model(inputs=x_input, outputs=y)
    #     model.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['accuracy'])
    #     model.summary()
    #     model.fit(X, Y, epochs=16, batch_size=64)
    #     if save:
    #         model.save("complex_model")
    #     test(model)
    #     return model

# def Create_neural_network4(training_data, save=False):
#         data = numpy.load(training_data)
#         X = data['X'].copy()
#         Y = data['Y'].copy()
#         x_input = k.Input((4, 1))
#         x = k.layers.Flatten()(x_input)
#         x = k.layers.Dense(20)(x)
#         x = k.layers.Activation('relu')(x)
#         x = k.layers.Dropout(rate=0.25)(x)
#         x = k.layers.Dense(50, activation='relu')(x)
#         x = k.layers.Dropout(rate=0.3)(x)
#         y = k.layers.Dense(1, activation='relu')(x)
#
#         model = Model(inputs=x_input, outputs=y)
#         model.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['accuracy'])
#         model.summary()
#         model.fit(X, Y, epochs=16, batch_size=64)
#         if save:
#             model.save("complex_model")
#         test(model)
#         return model


def Create_neural_network5(training_data, save=False):
    data = numpy.load(training_data)
    X = data['X'].copy()
    Y = data['Y'].copy()
    x_input = k.Input((4, 1))
    x = k.layers.Flatten()(x_input)
    x = k.layers.Dense(20, activation='relu')(x)
    x = k.layers.Dropout(rate=0.2)(x)
    x = k.layers.Dense(50)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Dropout(rate=0.25)(x)
    x = k.layers.Dense(50, activation='relu')(x)
    x = k.layers.Dropout(rate=0.3)(x)
    y = k.layers.Dense(1, activation='relu')(x)

    model = Model(inputs=x_input, outputs=y)
    model.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['accuracy'])
    model.summary()
    model.fit(X, Y, epochs=16, batch_size=64)
    if save:
        model.save("complex_model")
    test(model)
    return model


def test(model):
    pred = numpy.empty((2, 4, 1))
    pred[0, :, 0] = [4.102, 3, 3, 0]
    pred[1, :, 0] = [4.404, 6, 3, -1]
    print(pred)
    print(model.predict(pred))


Create_input_output_matrix()
# Create_neural_network("train_set_norm.npz", False)
# Create_neural_network1("train_set_norm.npz", False)
Create_neural_network5("train_set_norm.npz", False)
