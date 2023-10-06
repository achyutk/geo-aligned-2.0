Install the following packages if you want execute only these code, If the environment is already set up using the requirements file from the main directory, then this step can be skipped:

> pip install gensim <br>
> pip install python-Levenshtein <br>
> pip install nltk


### glove_to_word2vec.py
To execute this code:
> Download the [glove](https://nlp.stanford.edu/data/glove.6B.zip) model and save it in **/word2vec/model/glove.6B/** <br>
> Execute the python file. It will convert to word2vec file which is easily usable using gensim library for analogy task

### skip_gram_w2v_training.py

To execute this code replace the name of the dataset on *line 7*. It will pick the txt file from **/word2vec/data/** folder and create as skip-gram word2vec model in **word2vec/model** folder. 
Change the name of the word2vec file to the filename require at *line 26* 

### cbow_w2v_training.py

To execute this code replace the name of the dataset on *line 4*. It will pick the txt file from **/word2vec/data/** folder and create as skip-gram word2vec model in **word2vec/model** folder. 
Change the name of the word2vec file to the filename require at *line 19* 


### wiki_extraction.py

This code in **/word2vec/data/** is used to extract data from wikipedia. It creates two txt files which are stored in **/word2vec/data/**:
> link_list.txt : This file has list of links from wikipedia the code should extract the data from. <br>
> wiki_corpus.txt: This file contains the extracted data from wikipedia. This file is later used to train word2vec model. 

# wiki_extraction_india.py

This code in **/word2vec/data/** is used to extract data from wikipedia page of "India"/ It creates one file in the same location titled india_corpus.txt. This dataset is later used to train a word2vec model. 
