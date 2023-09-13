import nltk
import urllib
import bs4
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

# Extratcing list of links from wiki/List_of_countries_and_dependencies_by_population link
main = 'https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population'
source = urllib.request.urlopen(main).read()
soup=bs4.BeautifulSoup(source,'lxml')

country_links=[]
for link in soup.find_all('a'):
    url = link.get("href", "")
    if url.startswith("/wiki/") and '/' not in url[len("/wiki/"):]:
        country_links.append(url)
country_links=list(set(country_links))

# Extracting links present in each country file
links=[]
links=links + country_links

start= 'https://en.wikipedia.org'
corpus=[]
a=0
for i in country_links:
    source = urllib.request.urlopen(start+i).read()
    soup=bs4.BeautifulSoup(source,'lxml')
    link_2=[]
    for link in soup.find_all('a'):
        url = link.get("href", "")
        if url.startswith("/wiki/") and '/' not in url[len("/wiki/"):]:
            link_2.append(url)
    link_2=list(set(link_2))
    links = links +  link_2
    links = list(set(links))
    a=a+1

# Write the list to a file
with open('links_list.txt', 'w',encoding="utf-8") as file:
    for item in links:
        file.write(item + '\n')

# Reading Links file
f = open('links_list.txt', 'r',encoding="utf-8")
linkfrom_file=[]
for x in f.readlines():
    linkfrom_file.append(x)

#close the file
f.close()

links = linkfrom_file

corpus=[]
start= 'https://en.wikipedia.org'
for i in links:
    try:
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
    except:
        a=''

# Write the list to a file
with open('wiki_corpus.txt', 'w',encoding="utf-8") as file:
    for item in corpus:
        file.write(item + '\n')

