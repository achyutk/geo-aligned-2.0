import gensim
import pandas as pd

f = open('data/india_corpus.txt', 'r',encoding="utf-8") #Change the name of dataset wiki_corpus_txt to train on wikipedia dataset
sentences=[]
for x in f.readlines():
    sentences.append(x)

# close the file
f.close()

df= pd.DataFrame({'sentences':sentences})
review_text = df['sentences'].apply(gensim.utils.simple_preprocess)

model=gensim.models.Word2Vec(window=10,min_count=2,workers=4)
model.build_vocab(review_text,progress_per=1000)
model.train(review_text,total_examples=model.corpus_count,epochs=model.epochs)

model.save('./india_corpus_model.model') #Change the file name to wiki_corpus_model to save the model trained on wikipedia dataset
