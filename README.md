# Geo Stories

This software, accepts sentences in russian and generates an image which is geo-loacalised for indian audience. The framework developed has four components :
- Text Translation: Used for transalting sentences from russian to english. (The software uses:  OPUS mt ru-en)
- Text Mining: Identifying words which needs to be localised . (The software uses NERs, Parsing and WordNet)
- Word Embeddings: Used for aligning the words according to a geography by performing analogy. (The software uses Word2Vec/GloVe/BERT)
- Latent Diffusion Model: Used for generating an image. (The software uses fine-tuned stable diffusion model)
  
Below is the framework of this software:
![image](https://github.com/achyutk/geo-aligned-2.0/assets/73283117/f7ffccbf-7158-434d-8776-9fb9a13baa11)


# Installation
Clone repo and install requirements.txt in a Python>=3.8.0:

> git clone https://github.com/achyutk/geo-aligned-2.0.git     #clone <br>
> cd geo-aligned-2.0.git <br>
> pip install -r requirements.txt    #install <br>
> python -m spacy download en_core_web_sm    #Execute this for a specific spacy library <br>
> pip install -U git+https://github.com/openai/CLIP.git # Install this for evaluation 
 

# Model Weights 
Download the following model weight from the hyperlinks provided and paste it in the corresponding folder.

> Download [Word2Vec Embeddings](https://drive.google.com/drive/folders/1SGZEzirrWfHePQaDPVPhT-BzDWUWIkG8?usp=sharing) and place it into **/word2vec/model** folder. <br>
> Download the [Stable DIffusion (achyut\_sd)](https://drive.google.com/file/d/1eIkXxSf-3OodUtOONFM7F26wW4eBL9Mk/view?usp=sharing) and place it into **/diffusion_model** folder


# Datasets
The following datasets are used in this project:

> [India Corpus](https://drive.google.com/file/d/1_6bY8dqeg3I1-Rqtwrb1lhAX7Y6vGpF3/view?usp=sharing) : Used for training Word2Vec model <br>
> [Wikipedia Corpus](https://drive.google.com/file/d/1R4HeWvSDaxjFf2cysrxlwMT4_Msy79Dr/view?usp=sharing) : Used for training another Word2Vec model <br>
> [English Book Dataset](https://drive.google.com/drive/folders/1ZKY7XTQ6cQo1k8Lt24OQt0CGZ589Zekk?usp=sharing) : Used for evaluating the framework <br>
> [Russian Book Dataset](https://drive.google.com/drive/folders/1qVMe5ItBX6zgRyFy916eZEavVzgRUPuZ?usp=sharing) : Used for evaluating the framework


# Scripts

### main.ipynb

This file executes the framework. Make necessary changes for the framework in the model_download and utils files and run this jupyter notebook. 

Fill the text in the "sentence" variable for which the image is to be generated and run the remaining commands.

VOILA!!! The aligned image is generated

# Examples
Below is an example of results for different combinations of components for the fllowing example:

input in russian:  "Шерлок ест арбуз в Лондоне. Он одет в темную шляпу, накидку, четкую рубашку с высоким воротником, хорошо подогнанные брюки, начищенные туфли и жилет. Погода на улице дождливая"
input in english: "Sherlock eating watermelon in London. He is wearing a dark hat, cape, a crisp shirt with a high collar, well-fitted trousers, polished shoe and a waistcoat. Weather outside is rainy"

![Presentation - Achyut Karnani](https://github.com/achyutk/geo-aligned-2.0/assets/73283117/80077773-b025-4988-bd0e-0d52277628be)



# Further Reading 

Comming Soon...
