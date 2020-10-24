import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import keras as k
from keras.models import Model


def create_input_output_matrix():
    glove_input_file = 'glove.6B.50d.txt'
    word2vec_output_file = 'glove.6B.50d.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)

    # load the Stanford GloVe model
    filename = 'glove.6B.50d.txt.word2vec'
    model1 = KeyedVectors.load_word2vec_format(filename)

    file = open("pos and neg tweets.txt", "r", encoding='utf-8')
    lines = file.readlines()
    l = len(lines)

    # input of training data set
    X = np.empty((l, 50, 1))

    # output of training data set
    Y = np.empty((l, 1))

    # calculating sentence embedding  
    for i in range(0, l - 1):
        output = 0
        line = lines[i].replace("\n", "")
        data = line.split()
        pos_key = data[0][:5]
        if pos_key == "<pos>":
            data[0] = data[0].replace("<pos>", "")
            output = 1
        elif pos_key == "<neg>":
            data[0] = data[0].replace("<neg>", "")
            output = 0
        else:
            print("File format not supported\n")

        X[i, :, 0] = 0
        for j in range(0, len(data)):
            element = data[j].lower()
            if element in model1.vocab:
                X[i, :, 0] += model1[element]

        X[i, :, 0] = X[i, :, 0] / float(len(data))
        Y[i, 0] = output
    np.savez("pos_neg_sentence_embedding", X=X, Y=Y)


def create_neural_network(training_data, save=False):
    data = np.load(training_data)
    X = data['X'].copy()
    Y = data['Y'].copy()

    x_input = k.Input(shape=(50, 1))
    x = k.layers.Conv1D(filters=10, kernel_size=2, padding="same")(x_input)
    x = k.layers.AveragePooling1D(data_format="channels_last")(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Dense(200)(x)
    x = k.layers.Dropout(rate=0.3)(x)
    x = k.layers.Dense(75)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Dropout(rate=0.3)(x)
    x = k.layers.Dense(25)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Dropout(rate=0.3)(x)
    y = k.layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[x_input], outputs=y)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X, Y, epochs=8, batch_size=512)
    if save:
        model.save("basic_model")
    test(model)
    return model


def test(model):
    glove_input_file = 'glove.6B.50d.txt'
    word2vec_output_file = 'glove.6B.50d.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)

    # load the Stanford GloVe model
    filename = 'glove.6B.50d.txt.word2vec'
    model1 = KeyedVectors.load_word2vec_format(filename)

    file = open("pos and neg sample tweets.txt", "r", encoding='utf-8')
    lines = file.readlines()
    l = len(lines)
    pred = np.empty((l, 50, 1))

    for i in range(0, l - 1):
        line = lines[i].replace("\n", "")
        data = line.split()
        pred[i, :, 0] = 0
        for j in range(0, len(data)):
            element = data[j].lower()
            if element in model1.vocab:
                pred[i, :, 0] += model1[element]

        pred[i, :, 0] = pred[i, :, 0] / float(len(data))
        print(pred)
        print(model.predict(pred))


create_input_output_matrix()
create_neural_network("pos_neg_sentence_embedding.npz")
