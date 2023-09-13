import nltk
import urllib
import bs4
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/India').read()
soup=bs4.BeautifulSoup(source,'lxml')

text=" "
for paragraph in soup.find_all('p'):
    text+=paragraph.text

text=re.sub(r'\[[0-9]*\]','',text)
text= re.sub(r'\s+',' ',text)
text = text.lower()
text=re.sub(r'\d',' ',text)
text=re.sub(r'\s',' ',text)

sentences = nltk.sent_tokenize(text)

india_links=[]
for link in soup.find_all('a'):
    url = link.get("href", "")
    if url[:6]=="/wiki/":
        india_links.append(url)
india_links=list(set(india_links))

corpus=[]
corpus= corpus + sentences

start= 'https://en.wikipedia.org'

for i in india_links:
    source2 = urllib.request.urlopen(start+i).read()
    soup2=bs4.BeautifulSoup(source2,'lxml')

    text=" "
    for paragraph in soup2.find_all('p'):
        text+=paragraph.text

    text=re.sub(r'\[[0-9]*\]','',text)
    text= re.sub(r'\s+',' ',text)
    text = text.lower()
    text=re.sub(r'\d',' ',text)
    text=re.sub(r'\s',' ',text)
    sentences = nltk.sent_tokenize(text)
    corpus= corpus+sentences

# Write the list to a file
with open('india_corpus.txt', 'w',encoding="utf-8") as file:
    for item in corpus:
        file.write(item + '\n')