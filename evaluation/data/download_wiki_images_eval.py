import urllib.request
from PIL import Image
import pycountry
import os
import requests
from bs4 import BeautifulSoup


#FUnction to extract images for a country
def extract_image_links_from_wikipedia(page_title):
    base_url = "https://en.wikipedia.org/wiki/"
    full_url = base_url + page_title.replace(' ', '_')

    response = requests.get(full_url)

    if response.status_code != 200:
        print(f"Error: Unable to fetch the Wikipedia page '{page_title}'.")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    image_links = []

    for img_tag in soup.find_all('img'):
        image_src = img_tag.get('src')
        if image_src and (image_src.endswith('.jpg') or image_src.endswith('.jpeg')):
            image_links.append(image_src)

    return image_links

#Creating a list of country names
all_countries = list(pycountry.countries)
country_names = [country.name for country in all_countries]

#Iterating over the list of country name to download images
for i in country_names:
  page_title = i  # Replace this with the Wikipedia page title of your choice
  image_links = extract_image_links_from_wikipedia(i)

  for j in image_links:
    url = "https:" + j
    name= 'image_data/' +i+ '/' + url.split('/')[-1]
    print(name)
    if not os.path.exists('image_data/' +i):
      os.makedirs('image_data/' +i)
    urllib.request.urlretrieve(url,name)
    img = Image.open(name)