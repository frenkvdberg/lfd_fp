#!/usr/bin/env python

"""
Reads the data and preprocesses all the articles, replacing or
removing obvious give aways such as Newspaper names, author names,
emails, URLS and other strings that provide an unwanted shortcut in the classification task.
Authors: Esther Ploeger, Frank van den Berg
"""

from os import listdir  # to read files
from os.path import isfile, join  # to read files
import json
import re
from nltk import word_tokenize


def get_filenames_in_folder(folder):
    """Return all the filenames in a folder"""
    return [f for f in listdir(folder) if isfile(join(folder, f))]


def get_label(newspaper):
    """Get the political orientation label for a given newspaper
    or return False if newspaper not in dictionary"""
    polit_orientation = {"The Australian": "Right-Center",
                         "Sydney Morning Herald (Australia)": "Left-Center"}
    label = polit_orientation.get(newspaper, False)
    return label

def get_writer(raw_text):
    """Find out who wrote an article, to use in later cleaning"""
    writer = "Susan Owens"  # Using most common writer name, to make sure we always have a value
    for line in raw_text:
        if 'Byline' in line:
            writer = " ".join(line.split(" ")[1:3]).title()  # replace earlier value by actual writer
    return writer

def read_corpus(filename, tokens='word'):
    """Extract the data from the file, cleans the article and
    returns article text and political orientation label"""
    # Load the json file into a dictionary and store the articles
    data = json.load(open(filename, 'r'))
    articles = data["articles"]
    texts, labels = [], []

    # Go trough the articles to collect the tokenized texts and the PO labels
    for art in articles:
        label = get_label(art["newspaper"])
        if label:
            # Get body and check for a specific substring that only appears in advertisement articles
            body = art['body']
            body = body.replace("\n\n", " ")
            if 'More:' in body:
                continue

            # Replace or remove writing styles/habits that are specific to one newspaper":
            for x in ["MATP", " ...", "*", "-", "<", ">", "[", "]",
                      "â€¦", "______________________________"]:
                body = body.replace(x, "")
            for x in [" -", " -"]:
                body = body.replace(x, ",")
            body = body.replace('``', '"')
            body = body.replace('…', "")

            # Replace newspaper name, writer name and other giveaways
            writer = get_writer(art['raw_text'].split("\n"))
            body = body.replace("Sydney Morning Herald", "newspaper")
            body = re.sub(r'\bHerald\b(?! Square)', 'newspaper', body)  # Herald if not followed by the word Square
            body = re.sub(r'The Australian(?! [A-Z]| people)', '', body)  # Does not replace if followed by capital or 'people'
            body = body.replace(writer, "writer")  # replace name of the writer
            body = body.replace(writer.upper(), "writer")
            body = body.replace("writer reports. ", "")
            body = body.replace("AFP", "")
            body = re.sub(r'\S*@\S*\s?', '', body)  # remove emails
            body = body.replace("Comment on ALR stories at www.theaustralian.", "")
            body = body.replace("Read the full white paper at www.theaustralian.com.au", "")
            body = re.sub(r'Continued.*Page \d+.*From Page \d+', '', body)
            body = re.sub(r'Continued ([\w\s])*Page \d+', '', body)
            body = re.sub(r'Continued( on)? next page', '', body)
            body = re.sub(r'Continued( from)? previous page', '', body)
            body = re.sub(r'From previous page', '', body)
            body = re.sub(r'Page \d+', '', body)
            body = re.sub(r'Write to us .*$', '', body)
            body = re.sub(r'LETTERS TO THE EDITOR GPO Box .*$', '', body)
            body = re.sub(r'GPO Box .*$', '', body)
            body = re.sub(r'\.([\w\s])*writer is[^.]*\.', '.', body)
            body = re.sub(r'\.\.', '.', body)

            # Append text (either word- or sentence tokenized) and label to list
            if tokens == 'word':
                texts.append(word_tokenize(body))
            elif tokens == 'sent':
                t = word_tokenize(body)
                texts.append(" ".join(t).strip())
            labels.append(label)

    return texts, labels
