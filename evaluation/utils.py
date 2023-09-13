#!/usr/bin/env python
# coding: utf-8

# 1. Average Clip score of img wrt to 'aligned_input'
# 2. Average Clip score of img wrt to plain text- "india"
# 3. Average Clip score of 'aligned_input' wrt to plain text- "india"
# 4. Count the images whose highest clip score wrt different countries(plain text) is higher
# 5. Average similarity of images wrt to images from india wikipedia page.

import torch
import clip
from PIL import Image
import os
import pycountry
import pandas as pd
import numpy as np


# # Image to Aligned Text Similarity

def get_clip_score_img_text(model, preprocess,image_path, text):
    image = Image.open(image_path)
    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    try:
        text_input = clip.tokenize([text])
    except:
        text = text[:77]
        text_input = clip.tokenize([text])

    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()

    return clip_score


# Image to country text similarity

def get_clip_score_img_country(model,preprocess,image_path,text_feature_list,country):
        
    # Load the pre-trained CLIP model and the image
    image = Image.open(image_path)
    
     # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
           
    consine_list = [torch.matmul(image_features, text.to(device).half().T).item() for text in text_feature_list]       
    
    df= pd.DataFrame({'country':country,'cosines':consine_list})
    
    max_cosines = df['cosines'].idxmax()
    
    return df.loc[max_cosines]



# # Image to Image Similarity

def get_similarity_score_img_img(model, preprocess,image_path,country,img2_feature_list):
    
    # Load the pre-trained CLIP model and the image
    image = Image.open(image_path)
    
     # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    
    consine_list = [torch.matmul(image_features, img.to(device).half().T).item() for img in img2_feature_list]        
    
    df= pd.DataFrame({'country':country,'cosines':consine_list})
    
    avg_cosines = df.groupby(['country']).mean() 
    avg_cosines = avg_cosines.reset_index()
    
    max_cosines = avg_cosines['cosines'].idxmax()
    
    return avg_cosines.loc[max_cosines]

