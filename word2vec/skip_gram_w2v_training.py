from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import gensim

# Sample sentences for training
f = open('data/india_data.txt', 'r',encoding="utf-8")    #Change the file name to india_corpus_txt to train on indian dataset
sentences=[]
for x in f.readlines():
    sentences.append(x)

# close the file
f.close()

df= pd.DataFrame({'sentences':sentences})
review_text = df['sentences'].apply(gensim.utils.simple_preprocess)

# Preprocess sentences (tokenization)
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# Train CBOW model

model = Word2Vec(window=2,min_count=2,sg=1, workers=4) # This sg=1 indicates Skip Gram
model.build_vocab(review_text,progress_per=1000)
model.train(review_text,total_examples=model.corpus_count,epochs=model.epochs)
model.save("model/india_skip_gram_model")    #Change the file name to wiki_corpus_txt to train on wikipedia dataset

