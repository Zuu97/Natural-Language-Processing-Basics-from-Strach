import pandas as pd
import numpy as np
import pickle
import os
import re
import string
from variables import *
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

def GloveVectors():
    if not os.path.exists(word2vec_path):
        word2vec = {}
        with open(glove_path, encoding="utf8") as lines:
            for line in lines:
                line = re.split('[\n]', line)[0]
                line = line.split(' ')
                word, vec = line[0], line[1:]
                word2vec[word] = np.array(list(map(float,vec)))

        file_ = open(word2vec_path,'wb')
        pickle.dump(word2vec, file_)
        file_.close()
        print("Word2vec.pickle Saved!")
    else:
        print("Word2vec.pickle Loading!")
        file_ = open(word2vec_path,'rb')
        word2vec = pickle.load(file_)
        file_.close()
    return word2vec

def text2vector(line,word2vec):
    line = line.lower()
    tokens = line.split(' ')
    length = 0
    sum_vector = np.zeros(dim)
    for token in tokens:
        try:
            vec = word2vec[token]
            sum_vector += vec
            length +=  1
        except:
            pass
    return sum_vector/length

def preprocess_text_data(csv_path, word2vec, train=True):
    data = pd.read_csv(csv_path, header=None, sep='\t')
    data = data.dropna(axis = 0, how ='any')
    data.columns = ['class', 'content_text']
    classes = data['class']
    content = data['content_text']

    if train:
        encoder.fit(classes)
    Y = encoder.transform(classes) #list(le.inverse_transform([2, 2, 1])) decode back

    N = len(data)
    X = np.empty((N,dim))
    for n in range(N):
        line = content[n]
        X[n,:] = text2vector(line,word2vec)

    return X, Y

def get_data():
    word2vec = GloveVectors()
    Xtrain, Ytrain = preprocess_text_data(train_data_path, word2vec)
    Xtest, Ytest  = preprocess_text_data(test_data_path, word2vec, False)
    return Xtrain, Ytrain, Xtest, Ytest, encoder