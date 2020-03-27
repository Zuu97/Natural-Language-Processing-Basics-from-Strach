from gensim.models import KeyedVectors
from variables import googleNews_path

vectors = KeyedVectors.load_word2vec_format(
                            googleNews_path,
                            binary=True
                            )

def show_analogies(w1, w2, w3):
  r = vectors.most_similar(positive=[w1, w3], negative=[w2])
  print("%s - %s = %s - %s" % (w1, w2, r[0][0], w3))

def nearest_neighbors(w):
  r = vectors.most_similar(positive=[w])
  print("neighbors of: %s" % w)
  for word, score in r:
    print("\t%s" % word)


show_analogies('king', 'man', 'woman')
show_analogies('france', 'paris', 'london')
show_analogies('france', 'paris', 'rome')
show_analogies('paris', 'france', 'italy')
show_analogies('france', 'french', 'english')
show_analogies('japan', 'japanese', 'chinese')
show_analogies('japan', 'japanese', 'italian')
show_analogies('japan', 'japanese', 'australian')
show_analogies('december', 'november', 'june')
show_analogies('miami', 'florida', 'texas')
show_analogies('einstein', 'scientist', 'painter')
show_analogies('china', 'rice', 'bread')
show_analogies('man', 'woman', 'she')
show_analogies('man', 'woman', 'aunt')
show_analogies('man', 'woman', 'sister')
show_analogies('man', 'woman', 'wife')
show_analogies('man', 'woman', 'actress')
show_analogies('man', 'woman', 'mother')
show_analogies('heir', 'heiress', 'princess')
show_analogies('nephew', 'niece', 'aunt')
show_analogies('france', 'paris', 'tokyo')
show_analogies('france', 'paris', 'beijing')
show_analogies('february', 'january', 'november')
show_analogies('france', 'paris', 'rome')
show_analogies('paris', 'france', 'italy')

nearest_neighbors('king')
nearest_neighbors('france')
nearest_neighbors('japan')
nearest_neighbors('einstein')
nearest_neighbors('woman')
nearest_neighbors('nephew')
nearest_neighbors('february')
nearest_neighbors('rome')