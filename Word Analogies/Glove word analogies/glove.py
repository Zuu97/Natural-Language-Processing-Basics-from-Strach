from util import word_analogies, find_neighbours, word2vector

word2vec = word2vector()
def show_analogies(word1, word2, word3):
    analogy = word_analogies(word1, word2, word3, word2vec)
    print(word1+' - '+word2+' = '+analogy+' - '+word3)

def show_neighbours(word):
    print('neighbours for ',word,': ',find_neighbours(word, word2vec))

show_neighbours('king')
show_neighbours('france')
show_neighbours('japan')
show_neighbours('einstein')
show_neighbours('woman')
show_neighbours('nephew')
show_neighbours('february')
show_neighbours('rome')

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