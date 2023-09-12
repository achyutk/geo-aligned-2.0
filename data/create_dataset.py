# Importing necessary files

import re
import codecs
import os


# Fnction to read a txt file
def read_book_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        book_text = file.read()
    return book_text



# Function to merge paragraphs lines and remove newlines
def merge_lines_to_paragraphs(text):
    paragraphs = re.sub(r'\n+', ' ', text)
    return paragraphs



# Use regular expression to split the text into sentences
def split_sentences(paragraphs):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', paragraphs)
    return sentences



# Path for folder with books

path='C:/Users/ak19g21/Downloads/Project/data/english/book'    # For russian data
# path='C:/Users/ak19g21/Downloads/Project/data/russian/book'    # For russian data Uncomment for english data extraction


book_list=os.listdir(path)  # List to store the names of books
data=[]    # List to save the data from all the books

# Iterating over books
for i in book_list:
    file_path = path+'/'+i  # Replace with the actual path to your book file
    book_text = read_book_file(file_path) #Reading the book
    paragraphs = merge_lines_to_paragraphs(book_text) #Merging sentences/paragraphsn the book
    sentences = split_sentences(paragraphs)   # extracting sentences from the book
    sentences=sentences[50:-50]  # Ignoring the first and last 50 lines of the books which generally have index 
    sentences = [x for x in sentences if len(x.split(' '))>3]   # Ignoring sentence with length less than 3
    data=data+sentences



with open('data.txt', 'w',encoding="utf-8") as file:
    for item in data:
        file.write(item + '\n')

