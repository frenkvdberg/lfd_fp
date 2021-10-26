#!/usr/bin/env python

"""

"""

from os import listdir  # to read files
from os.path import isfile, join  # to read files
import json
import re
from nltk import sent_tokenize, word_tokenize, pos_tag


def get_filenames_in_folder(folder):
	"""
	return all the filenames in a folder
	"""
	return [f for f in listdir(folder) if isfile(join(folder, f))]


def get_label(newspaper):
	""""""
	polit_orientation = {"The Australian": "Right-Center",
	                     "Sydney Morning Herald (Australia)": "Left-Center"}

	label = polit_orientation.get(newspaper, False)
	return label


def read_corpus(filename, tokens='word'):
	"""x"""
	texts = []
	labels = []

	# Load the json file into a dictionary
	data = json.load(open(filename, 'r'))

	# Store the articles and define list of stopwords:
	articles = data["articles"]

	# Go trough the articles to collect the tokenized texts and the PO labels
	for art in articles:
		label = get_label(art["newspaper"])
		if label:
			# Find who wrote it:
			writer = "Susan Owens" # Using most common writer name, to make sure we always have a value
			rt = art['raw_text'].split("\n")
			for line in rt:
				if 'Byline' in line:
					writer = " ".join(line.split(" ")[1:3]).title()  # replace earlier value by actual writer

			# Clean text
			body = art['body']
			body = body.replace("\n\n", " ")

			if 'More:' in body:
				continue  # only advertisement articles contain this substring

			# Replace or remove writing styles/habits that are specific to one newspaper":
			#body = re.sub(r'Continued.*--.*Page \d+ From Page \d+', '', body)
			for x in ["MATP", " ...", "*", "-", "<", ">", "[", "]",
					  "â€¦", "______________________________"]:
				body = body.replace(x, "")
			for x in [" -", " -"]:
				body = body.replace(x, ",")
			body = body.replace('``', '"')
			body = body.replace('…', "")

			# Replace newspaper name, writer name and other giveaways
			body = body.replace("Sydney Morning Herald", "newspaper")
			body = re.sub(r'\bHerald\b(?! Square)', 'newspaper', body)  # Herald if not followed by the word Square
			body = re.sub(r'The Australian(?! [A-Z]| people)', '', body)  # Does not replace if followed by capital or 'people'
			body = body.replace(writer, "writer") # replace name of the writer
			body = body.replace(writer.upper(), "writer")  # replace name of the writer
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
				texts.append(sent_tokenize(body))
			labels.append(label)

	return texts, labels
