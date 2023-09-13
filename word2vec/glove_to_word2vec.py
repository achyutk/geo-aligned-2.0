from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

#Code will convert the glve model to word2vec model
glove_input_file = 'C:/Users/ak19g21/Downloads/Project/word2vec/model/glove.6B/glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)


# Testing the glove_w2v model for application

# load the Stanford GloVe model
filename = 'glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)

