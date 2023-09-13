import torch
import clip
from PIL import Image
import os
import numpy as np
import pandas as pd
import csv

def get_clip_score_img_features(model, preprocess,image_path):
    image = Image.open(image_path)

    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)

    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    model = model.to(device)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features

# Load the pre-trained CLIP model and the image
model, preprocess = clip.load('ViT-B/32')

features=[] # List to store all the features
country=[] # List to store names of countries
path='image_data'

folders = os.listdir(path) # List to store country names

# Iterating over folders of countries
for i in folders:
    files= os.listdir(path+'/'+i)

    # Iterating over images
    for j in files:
        image_path = path + '/' + i + '/' + j # path of image

        # Extracting features
        img_feature = get_clip_score_img_features(model,preprocess,image_path)

        # Appending values
        features.append(img_feature)
        country.append(i.split('_')[0])

# storing the list of tensor the features in csv file
torch.save(features, 'eval_img_features.pt')

# Save the list of strings to a text file
with open('eval_img_features_country_list.txt', 'w') as file:
    for string in country:
        file.write(string + '\n')