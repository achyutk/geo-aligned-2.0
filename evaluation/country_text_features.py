import torch
import clip
from PIL import Image
import os
import numpy as np
import pandas as pd
import csv
import pycountry

def get_clip_text_features(model, preprocess, text_list):

    feature=[]

    for text in text_list:
        text_input = clip.tokenize([text])

        # Move the inputs to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_input = text_input.to(device)
        model = model.to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_input)

        # Normalize the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        feature.append(text_features)

    return feature



# Load the pre-trained CLIP model and the image
model, preprocess = clip.load('ViT-B/32')

#Creating a list of names of the countries
all_countries = list(pycountry.countries)
country_names = [country.name for country in all_countries]

#Extracting features for the names of the country
features = get_clip_text_features(model, preprocess,country_names)

# storing the list of tensor the features in csv file
torch.save(features, 'model/eval_country_text_features.pt')

# Save the list of strings to a text file
with open('model/eval_country_text_country_list.txt', 'w') as file:
    for string in country_names:
        file.write(string + '\n')