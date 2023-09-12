# -*- coding: utf-8 -*-
""" framework.ipynb """
# Importing necessary Libraries
# from huggingface_hub import notebook_login
# notebook_login()
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from gensim.models import KeyedVectors
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import wordnet
import spacy
nltk.download('wordnet')
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")

# Performing Translation
def translation(model,tokenizer,text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated



def convert_entities_to_list(text, entities: list[dict]) -> list[str]:
        ents = []
        for ent in entities:
            e = {"start": ent["start"], "end": ent["end"], "label": ent["entity_group"]}
            if ents and -1 <= ent["start"] - ents[-1]["end"] <= 1 and ents[-1]["label"] == e["label"]:
                ents[-1]["end"] = e["end"]
                continue
            ents.append(e)

        return [text[e["start"]:e["end"]] for e in ents]

def convert_entities_to_list2(text, entities: list[dict]) -> list[str]:
        ents = []
        for ent in entities:
            e = {"start": ent["start"], "end": ent["end"], "label": ent["entity"]}
            if ents and -1 <= ent["start"] - ents[-1]["end"] <= 1 and ents[-1]["label"] == e["label"]:
                ents[-1]["end"] = e["end"]
                continue
            if ent['entity']!='B-TIME':
                ents.append(e)

        return [text[e["start"]:e["end"]] for e in ents]

def other_entities(text):
    nlp = spacy.load('en_core_web_sm')
    parsed=nlp(text.lower())
    parsed_sentence= [(x.text,x.pos_,x.dep_,[(x.text,x.dep_) for x in list(x.children)]) for x in parsed]
    original_words = [x[0] for x in parsed_sentence]
    words = [x[0] for x in parsed_sentence if x[1]=='ADJ']
    words= words + [x[0] for x in parsed_sentence if x[1]=='NOUN']
    words= words + [x[0] for x in parsed_sentence if x[1]=='PROPN']
    words= words + [x[0] for x in parsed_sentence if x[1]=='ADJ']
    words= words + [x[0] for x in parsed_sentence if x[2]=='POBJ']
    words= list(set(words))
    words = sorted(words, key=lambda x: original_words.index(x))

    return original_words,words


# Performing NER identification
def ner_identification(food_tokenizer,food_model,loc_per_tokenizer,loc_per_model,clothes_tokenizer,clothes_model,text):

    ###Food identification
    food_pipe = pipeline("ner", model=food_model, tokenizer=food_tokenizer)
    food_ner_results = food_pipe(text, aggregation_strategy="simple")

    ### Animal identification
    # Tokenize the sentence into words
    words = nltk.word_tokenize(text)

    # Initialize an empty list to store detected animal names
    animals_plants = []

    # Iterate over each word and check if it is an animal
    for word in words:
        # Get the synsets (set of synonyms) for the word
        synsets = wordnet.synsets(word)
        # Check if any synset is for an animal (noun synset)
        animal_synsets = [synset for synset in synsets if synset.pos() == 'n' and 'animal' in synset.lexname()]
        plant_synsets = [synset for synset in synsets if synset.pos() == 'n' and 'plants' in synset.lexname()]

        # If animal synsets are found, add the word to the animals list
        if animal_synsets:
            animals_plants.append(word)
        if plant_synsets:
            animals_plants.append(word)

    ###Location and Person identification
    loc_per_pipe = pipeline("ner", model=loc_per_model, tokenizer=loc_per_tokenizer)
    loc_per_ner_results = loc_per_pipe(text)

    ###Clothes identification
    clothes_pipe = pipeline("ner", model=clothes_model, tokenizer=clothes_tokenizer)
    clothes_ner_results = clothes_pipe(text)

    identified_words= convert_entities_to_list(text, food_ner_results) + animals_plants + convert_entities_to_list2(text, loc_per_ner_results) + convert_entities_to_list2(text, clothes_ner_results)

    identified_words =[x.lower() for x in identified_words]

    return identified_words

# Performing Geo-alignment
def alignment(word2vec_model,identified_words,aligned_sentence):
    replace=[]
    for i in identified_words:
        try:
            result = word2vec_model.most_similar(positive=[i,'india'], topn=1)
            score1 = word2vec_model.similarity(result[0][0],i)
            score2 = word2vec_model.similarity(result[0][0],'india')
            if score1>0 and result[0][1]>0:
                replace.append(result[0][0])
            else:
                replace.append('Nan')
        except:
            replace.append('Nan')

    for i in range(len(identified_words)):
        if replace[i]!='Nan':
            aligned_sentence = aligned_sentence.replace(identified_words[i], replace[i],1)


    return replace,aligned_sentence

# Performing image-generation
def generate_image(disffusion_model,aligned_sentence):

    ''' Below code is for using pre-trained diffusion model by nitrosocke'''

    aligned_sentence = aligned_sentence + '. modern disney style'
    image= disffusion_model(aligned_sentence).images[0]


    ''' Below code is for using pre-trained diffusion model by Achyut'''
    
    # aligned_sentence = 'In indo_style.' + aligned_sentence
    # image= disffusion_model(aligned_sentence).images[0]


    return image


"""# Main"""

def main(translation_model,translation_tokenizer, food_tokenizer,food_model,loc_per_tokenizer,loc_per_model,clothes_tokenizer,clothes_model,word2vec_model,disffusion_model,original_text):

    # Performing translations
    translated_text = translation(translation_model,translation_tokenizer,original_text)
    
    # Identifying entities using parsing
    original_words,words = other_entities(translated_text)
    
    # Creating a copy of translated text for alignment
    aligned_sentence= ''+translated_text
    aligned_sentence = aligned_sentence.lower()
    
    # Identifying entities using ner
    identified_words= ner_identification(food_tokenizer,food_model,loc_per_tokenizer,loc_per_model,clothes_tokenizer,clothes_model,translated_text)

    # Combining the list of identified entities
    identified_words = identified_words + words
    # Removing duplicates of entities
    identified_words = list(set(identified_words))

    # Sorting the entities based on their appearance in the sentence   
    # identified_words = sorted(identified_words, key=lambda x: original_words.index(x))
    identified_words = sorted(identified_words, key=lambda x: (original_words.index(x) if x in original_words else len(original_words)))


    replace,aligned_sentence = alignment(word2vec_model,identified_words,aligned_sentence)
    image = generate_image(disffusion_model,aligned_sentence)

    return translated_text, aligned_sentence , image


def main_no_translation(translation_model,translation_tokenizer, food_tokenizer,food_model,loc_per_tokenizer,loc_per_model,clothes_tokenizer,clothes_model,word2vec_model,disffusion_model,original_text):
    
        
    aligned_sentence= ''+original_text
    aligned_sentence = aligned_sentence.lower()
    original_words,words = other_entities(aligned_sentence)
    
    identified_words= ner_identification(food_tokenizer,food_model,loc_per_tokenizer,loc_per_model,clothes_tokenizer,clothes_model,original_text)
    identified_words = identified_words + words
    identified_words = list(set(identified_words))
    identified_words = filter(lambda x: x in original_words, identified_words)
    # identified_words = sorted(identified_words, key=lambda x: original_words.index(x))
    identified_words = sorted(identified_words, key=lambda x: (original_words.index(x) if x in original_words else len(original_words)))
    
    replace,aligned_sentence = alignment(word2vec_model,identified_words,aligned_sentence)
 
    
    image = generate_image(disffusion_model,aligned_sentence)
    
    return aligned_sentence , image