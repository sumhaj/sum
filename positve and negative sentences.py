import numpy as np
import keras as k
from keras.models import Model



def Create_input_output_matrix():
    # putting all positive and negative words in a dictionary
    posi = open("positive words list.txt", "r")
    word1 = posi.readlines()
    l = len(word1)
    pos_word = {}
    conj_word = {}

    for i in range(0, l):
        word1[i] = word1[i].replace(" ", "")
        word1[i] = word1[i].replace("\n", "")
        pos_word[word1[i].upper()] = True

    negi = open("negative words list.txt", "r")
    word2 = negi.readlines()
    l1 = len(word2)

    for i in range(0, l1):
        word2[i] = word2[i].replace(" ", "")
        word2[i] = word2[i].replace("\n", "")
        pos_word[word2[i].upper()] = False

    # putting conjunctions in a dictionary
    conj = open("conjunctions list.txt", "r")
    word = conj.readlines()
    l2 = len(word)

    for i in range(0, l2):
        word[i] = word[i].replace(" ", "")
        word[i] = word[i].replace("\n", "")
        conj_word[word[i].upper()] = 1

    # reading pos and neg sentences
    file = open("pos and neg tweets.txt", "r", encoding='utf-8')
    lines = file.readlines()


    # input of training data set
    X = np.empty((len(lines), 3, 1))

    # output of training data set
    Y = np.empty((len(lines), 1))

    # for key,value in conj_word.items():
    #     print(key, ":", value)


        # calculating features (positive count, negative count and positive to negative count ratio)
    for i in range(0, len(lines)-1):
        pos_count = 0
        neg_count = 0
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

        for j in range(0, len(data)):
            element = data[j].upper()

            if element in conj_word:
                pos_count = 0
                neg_count = 0

            if element in pos_word:
                if pos_word[element]:
                    pos_count = pos_count+1
                elif not pos_word[element]:
                    neg_count = neg_count+1

        nc = neg_count
        if neg_count == 0:
            nc = 1
        pnr = float(pos_count)/float(nc)


        # putting values of features in empty input output array
        X[i, :, 0] =[pos_count, neg_count, pnr]
        Y[i, 0] = output

    np.savez("pos_neg_training_data", X=X, Y=Y)


# def Create_neural_network(training_data, save=False):
#     data = np.load(training_data)
#     X = data['X'].copy()
#     Y = data['Y'].copy()
#
#     x_input = k.Input((3, 1))
#     x = k.layers.Flatten()(x_input)
#     x = k.layers.Dense(3)(x)
#     x = k.layers.Activation('softmax')(x)
#     x = k.layers.Dropout(rate=0.3)(x)
#     y = k.layers.Dense(1, activation='softmax')(x)
#
#     model = Model(inputs=x_input, outputs=y)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
#     model.fit(X, Y, epochs=8, batch_size=256)
#     if save:
#         model.save("basic_model")
#     test(model)
#     return model


def Create_neural_network1(training_data, save=False):
    data = np.load(training_data)
    X = data['X'].copy()
    Y = data['Y'].copy()

    x_input = k.Input((3, 1))
    x = k.layers.Flatten()(x_input)
    x = k.layers.Dense(10)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Dropout(rate=0.3)(x)
    x = k.layers.Dense(20)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Dropout(rate=0.25)(x)
    y = k.layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=x_input, outputs=y)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X, Y, epochs=8, batch_size=256)
    if save:
        model.save("basic_model")
    test(model)
    return model


def test(model):
    posi = open("positive words list.txt", "r")
    word1 = posi.readlines()
    l = len(word1)
    pos_word = {}
    conj_word = {}

    for i in range(0, l):
        word1[i] = word1[i].replace(" ", "")
        word1[i] = word1[i].replace("\n", "")
        pos_word[word1[i].upper()] = True

    negi = open("negative words list.txt", "r")
    word2 = negi.readlines()
    l1 = len(word2)

    for i in range(0, l1):
        word2[i] = word2[i].replace(" ", "")
        word2[i] = word2[i].replace("\n", "")
        pos_word[word2[i].upper()] = False

    # putting conjunctions in a dictionary
    conj = open("conjunctions list.txt", "r")
    word = conj.readlines()
    l2 = len(word)

    for i in range(0, l2):
        word[i] = word[i].replace(" ", "")
        word[i] = word[i].replace("\n", "")
        conj_word[word[i].upper()] = 1

    file = open("pos and neg tweets sample.txt", "r")
    lines = file.readlines()

    pred = np.empty((6, 3, 1))

    for i in range(0, len(lines)):
        pos_count = 0
        neg_count = 0
        line = lines[i].replace("\n", "")
        data = line.split()


        for j in range(0, len(data)):
            element = data[j].upper()

            if element in conj_word:
                pos_count = 0
                neg_count = 0

            if element in pos_word:
                if pos_word[element]:
                    pos_count = pos_count + 1
                elif not pos_word[element]:
                    neg_count = neg_count + 1

        nc = neg_count
        if (neg_count == 0):
            nc = 1
        pnr = float(pos_count) / float(nc)
        pred[i, :, 0] = [pos_count, neg_count, pnr]
    print(pred)
    print(model.predict(pred))


Create_input_output_matrix()
# Create_neural_network("pos_neg_training_data.npz", False)
Create_neural_network1("pos_neg_training_data.npz", True)






