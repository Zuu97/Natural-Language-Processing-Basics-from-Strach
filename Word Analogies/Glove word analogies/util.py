from variables import*
import numpy as np
import pickle
import os
import re
from sortedcontainers import SortedDict

def word2vector():
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

def cosine_similarity(v1, v2):
    cosine_value = np.dot(v1.T, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return (1 - cosine_value)

def find_neighbours(word1, word2vec, neighbours=5):
    vec1 = word2vec[word1]
    sorted_dict = SortedDict()
    for word2,vec2 in word2vec.items():
        cosine_sim = cosine_similarity(vec1,vec2)
        if not (vec1 == vec2).all():
            if len(sorted_dict) < neighbours:
                sorted_dict[cosine_sim] = word2
            else:
                dis0 = list(sorted_dict.keys())[-1]
                if dis0 > cosine_sim:
                    del sorted_dict[dis0]
                    sorted_dict[cosine_sim] = word2

    return list(sorted_dict.values())

def find_analogies(vec, word2vec):
    dist2word= {cosine_similarity(vec,veci):word for word, veci in word2vec.items()}
    shortest_dist = min(list(dist2word.keys()))
    return dist2word[shortest_dist]

def word_analogies(word1,word2,word3,word2vec):
    vec1 = word2vec[word1]
    vec2 = word2vec[word2]
    vec3 = word2vec[word3]

    target_vec = vec1 - vec2 + vec3
    return find_analogies(target_vec, word2vec)
