Clone repo and install requirements.txt in a Python>=3.8.0:

> git clone https://github.com/achyutk/geo-aligned-2.0.git     #clone <br>
> cd geo-aligned-2.0.git <br>
> pip install -r requirements.txt    #install
> python -m spacy download en_core_web_sm    #Execute this for a specific spacy library
> pip install -U git+https://github.com/openai/CLIP.git # Install this for evaluation
 


Download the following model weight from the hyperlinks provided and paste it in the corresponding folder.

> Download [Word2Vec Embeddings](https://drive.google.com/drive/folders/1SGZEzirrWfHePQaDPVPhT-BzDWUWIkG8?usp=sharing) and place it into **/word2vec/model** folder. <br>
> Download the [Stable DIffusion (achyut\_sd)](https://drive.google.com/file/d/1eIkXxSf-3OodUtOONFM7F26wW4eBL9Mk/view?usp=sharing) and place it into **/diffusion_model** folder



The following datasets are used in this project:

> [India Corpus](https://drive.google.com/file/d/1_6bY8dqeg3I1-Rqtwrb1lhAX7Y6vGpF3/view?usp=sharing) : Used for training Word2Vec model <br>
> [Wikipedia Corpus](https://drive.google.com/file/d/1R4HeWvSDaxjFf2cysrxlwMT4_Msy79Dr/view?usp=sharing) : Used for training another Word2Vec model <br>
> [English Book Dataset](https://drive.google.com/drive/folders/1ZKY7XTQ6cQo1k8Lt24OQt0CGZ589Zekk?usp=sharing) : Used for evaluating the framework <br>
> [Russian Book Dataset](https://drive.google.com/drive/folders/1qVMe5ItBX6zgRyFy916eZEavVzgRUPuZ?usp=sharing) : Used for evaluating the framework



# main.ipynb

This file executes the framework. Make necessary changes for the framework in the model_download and utils files and run this jupyter notebook. 

Fill the text in the "sentence" variable for which the image is to be generated and run the remaining commands.

VOILA!!! The aligned image is generated



