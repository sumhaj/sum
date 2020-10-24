import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

# load the Stanford GloVe model
filename = 'glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename)

file = open("sentences.txt", "r")
lines = file.readlines()
l = len(lines)
vector = np.empty((l, 50, 1))

for i in range(0, l):
    line = lines[i].replace("\n", "")
    data = line.split()
    vector[i, :, 0] = 0
    for j in range(0, len(data)):
        element = data[j].lower()
        vector[i, :, 0] += model[element]

    vector[i, :, 0] = vector[i, :, 0] / float(len(data))
np.save("sentence_embedding", vector)

c = 0
sum1 = 0
sum2 = 0
for i in range(0, 50):
    c += vector[0, i, 0] * vector[1, i, 0]
    sum1 += abs(vector[0, i, 0])
    sum2 += abs(vector[1, i, 0])
cosine = c/float((sum1 * sum2) ** 0.5)
print("similarity: ", cosine)