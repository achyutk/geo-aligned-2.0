from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from gensim.models import KeyedVectors

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM

import nltk
from nltk.corpus import wordnet
import spacy
nltk.download('wordnet')
nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")


def download_translation():
    mname = 'Helsinki-NLP/opus-mt-ru-en'
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
    return model,tokenizer

def download_ner():
    # Food NER
    food_tokenizer = AutoTokenizer.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
    food_model = AutoModelForTokenClassification.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
    # Food Location and person NER
    loc_per_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    loc_per_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    # Clothes NER
    clothes_tokenizer = AutoTokenizer.from_pretrained("CouchCat/ma_ner_v7_distil")
    clothes_model = AutoModelForTokenClassification.from_pretrained("CouchCat/ma_ner_v7_distil")  
    return food_tokenizer,food_model,loc_per_tokenizer,loc_per_model,clothes_tokenizer,clothes_model


def download_word2vec():

    path='word2vec/model/'
    
    '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
    ''' Below code is for using pre-trained  Word2Vec'''
    
    filename = 'GoogleNews-vectors-negative300.bin.gz'
    word2vec_model = KeyedVectors.load_word2vec_format(path+filename, binary=True)
  

    ''' Below code is for using trained Word2Vec'''
    
    # filename = 'india_corpus_model.model'
    # filename = 'wiki_corpus_model.model'
    # word2vec = KeyedVectors.load(path+filename, mmap='r')
    # word2vec_model = word2vec.wv
    '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
    ''' Below code is for using pre-trained  Glove'''
    
    # filename = 'glove.6B.50d.txt.word2vec'
    # word2vec_model = KeyedVectors.load_word2vec_format(path+filename, binary=False)

    return word2vec_model


def download_stable_diffusion():
    

    ''' Below code is for using pre-trained diffusion model by nitrosocke'''

    torch.manual_seed(1000)
    model_id = "nitrosocke/mo-di-diffusion"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    disffusion_model = pipe.to('cuda')
    
    '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
    
    ''' Below code is for using pre-trained diffusion model by Achyut'''
    # model_path = "C:/Users/ak19g21/Downloads/Project/diffusion_model/model.ckpt"
    # torch.manual_seed(1000)
    # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # pipe = StableDiffusionPipeline.from_single_file(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)
    # disffusion_model = pipe.to('cuda')
    
    return disffusion_model


def download_models():
  
    translation_model,translation_tokenizer=download_translation() # translation models
    food_tokenizer,food_model,loc_per_tokenizer,loc_per_model,clothes_tokenizer,clothes_model = download_ner() # ner models
    word2vec_model = download_word2vec() # word2vecmodel for alignment
    disffusion_model = download_stable_diffusion() # stable_diffusion_download
    
    return translation_model,translation_tokenizer, food_tokenizer,food_model,loc_per_tokenizer,loc_per_model,clothes_tokenizer,clothes_model,word2vec_model,disffusion_model