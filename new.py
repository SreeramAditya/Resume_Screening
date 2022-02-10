#! /usr/bin/python3

from tkinter import Scrollbar
import PySimpleGUI as gui
import time
import os

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
import re
import json
import pickle

from keras.models import load_model
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english')+['``',"''"])


def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    resumeText = resumeText.lower()  # convert to lowercase
    resumeTextTokens = word_tokenize(resumeText)  # tokenize
    filteredText = [w for w in resumeTextTokens if not w in stopwords_set]  # remove stopwords
    return ' '.join(filteredText)

def predict_scores(input) :

	#print("function predict_scores")
	#return

	# load model
	model = load_model(r'C:\Users\home\Desktop\model')

	# convert input to vectorized format
	input = cleanResume(input)

	max_length = 300
	trunc_type = 'post'
	padding_type = 'post'

	# Get feature text tokenizer used for model training
	with open(r'assets/tokenizer/feature_tokenizer.pickle', 'rb') as handle:
		feature_tokenizer = pickle.load(handle)

	# Get label encoding dictionary from model training
	with open(r'assets/dictionary/dictionary.pickle', 'rb') as handle:
		encoding_to_label = pickle.load(handle)

	# Handle unknown label case and load original labels
	encoding_to_label[0] = 'unknown'
	with open(r"assets/data/labels.json", "r") as read_file:
		original_labels = json.load(read_file)

	predict_sequences = feature_tokenizer.texts_to_sequences([input])
	predict_padded = pad_sequences(predict_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
	predict_padded = np.array(predict_padded)

	# predict the result with vectored input
	prediction = model.predict(predict_padded)

	# modify and print the output
	encodings = np.argpartition(prediction[0], -5)[-5:]
	encodings = encodings[np.argsort(prediction[0][encodings])]
	encodings = reversed(encodings)

	for encoding in encodings:
		label = encoding_to_label[encoding]
		probability = prediction[0][encoding] * 100
		probability = round(probability, 2)
		gui.Print('{} - {}%'.format(original_labels[label], probability))



gui.theme("Default")
_font = "Times New Roman"
_font_size = 12
start_time = int(round(time.time() * 100))
current_time = 0
limit = 1

layout = [
			#[ gui.InputText(default_text = "Untitled", key = 'file_name', ),
			#gui.Combo(key = "font",values=['Times New Roman', 'Arial', 'Calbri'], size=(20, 6), default_value=_font, auto_size_text=True, enable_events=True, change_submits=True),
			#gui.Combo(key = "font_size",values=[12, 13, 14, 15, 16, 17, 18, 19, 20], size = (3, 5), default_value=_font_size, enable_events=True, change_submits=True),
			#gui.Text(text = "00:00:00", key = "timer", size = (9, 1))
			#],
			[gui.Multiline(key = "content", font = (_font, _font_size), enable_events=True, autoscroll=True, auto_refresh=True, change_submits = True, size = (70,30), no_scrollbar = True)],
			[gui.B("Predict", key = "predict"), gui.B("Reset", key = "reset"), gui.B("Close", key = "close")]

	]

window = gui.Window('Resume-Screener', layout, resizable = False)

while True:
	event, values = window.read()
	# current_time = int(round(time.time() * 100)) - start_time

	# window['timer'].update('{:02d}:{:02d}.{:02d}'.format((current_time // 100) // 60,
  #                                                      (current_time // 100) % 60,
     #                                                   current_time % 100))

	if event == gui.WIN_CLOSED or event == 'close':
		# gui.Popup("It is auto-saved")
		break

	if event == 'font' :
		_font = values['font']
		window["content"].update(font = (_font, _font_size))

	if event == "font_size" :
		_font_size = values['font_size']
		window["content"].update(font = (_font, _font_size))

	if event == "reset" :
		#values["content"] = ""
		window["content"].update(value = "")

	if event == "predict" :
		# values[content] --> the content in text field
		# fxn to predict
		# Print("Predict Button")

		inp = values["content"].strip()

		if len(inp):
			# print(inp)
			predict_scores(inp)

		else :
			gui.Popup("Content is Empty. \n Please enter the appropriate input.")


window.close()
