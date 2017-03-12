import pandas as pd
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import argparse

parser = argparse.ArgumentParser(description='Choose model parameters')
parser.add_argument('--preprocessed', type=bool, default=False,
	help='loads preprocessed corpus')

args = parser.parse_args()
preprocessed = args.preprocessed

def load_corpus(path, usecols=['gold_label', 'sentence1_binary_parse', 'sentence2_binary_parse']):
	df = pd.read_csv(path,
		delimiter='\t', usecols=usecols)
	return df

def drop_unclassified(df):
    #Drop the samples without gold_label
    df = df[df.gold_label != '-']
    return df

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

labels_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

def classes_to_categories(df):
	df.loc[df['gold_label'] == 'entailment', 'gold_label'] = labels_dict['entailment']
	df.loc[df['gold_label'] == 'contradiction', 'gold_label'] = labels_dict['contradiction']
	df.loc[df['gold_label'] == 'neutral', 'gold_label'] = labels_dict['neutral']

def categories_to_categorical(labels, nb_classes):
	return to_categorical(labels, nb_classes=nb_classes)

def tokens_to_sentences(tokenized_sentence1, tokenized_sentence2):
	sent1 = [' '.join(x) for x in tokenized_sentence1]
	sent2 = [' '.join(x) for x in tokenized_sentence2]
	return sent1, sent2	

"""
Preprocessing data
1. Load dataframe
2. Drop samples unclassified (gold_label == '-')
3. Convert labels to integers, according to labels_dict
4.Extract tokens from binary parse
5. Convert tokens to sentences
6. Build a binary class matrix representation of the integer labels
7.Convert sentences to index sequences, according to tokenizer.word_index
8. Pad sequences, so they all have the same length MAX_LEN (MAX_LEN = 82 (default)) 

"""

if not preprocessed:
	print ("Preprocessing corpus...")
	path = 'snli_1.0/snli_1.0_train.txt'
	df = load_corpus(path)
	df = drop_unclassified(df)
	classes_to_categories(df)
	df['sentence1_binary_parse'] = df['sentence1_binary_parse'].apply(
		lambda x: extract_tokens_from_binary_parse(str(x)))
	df['sentence2_binary_parse'] = df['sentence2_binary_parse'].apply(
		lambda x: extract_tokens_from_binary_parse(str(x)))
	# print (max(len(x) for x in df['sentence1_binary_parse'])) #	82 
	# print (max(len(x) for x in df['sentence2_binary_parse'])) #	62
	MAX_LEN = 82

	tokenizer = Tokenizer(lower=False, filters='')
	sent1, sent2 = tokens_to_sentences(df['sentence1_binary_parse'], 
		df['sentence2_binary_parse'])
	tokenizer.fit_on_texts(sent1 + sent2)
	# print (len(tokenizer.word_counts))#	42389 unique tokens

	y = categories_to_categorical(df['gold_label'], 3)
	# print y[0]

	to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
	sentences_to_padded_index_sequences = lambda data: to_seq(data)

	sent1, sent2 = sentences_to_padded_index_sequences(sent1), \
	sentences_to_padded_index_sequences(sent2)
	# print ('Shape of sent1 tensor: ', sent1.shape)#	('Shape of sent1 tensor: ', (549367, 82))
	# print ('Shape of sent2 tensor: ', sent2.shape)#	('Shape of sent2 tensor: ', (549367, 82))
	# print ('Shape of label tensor: ', y.shape)#	('Shape of label tensor: ', (549367, 3))

	np.save("snli_train_sent1.npy", sent1, allow_pickle=False)
	np.save("snli_train_sent2.npy", sent1, allow_pickle=False)
	np.save("snli_train_labels.npy", y, allow_pickle=False)

	if os.path.exists("snli_train_sent1.npy") and \
	os.path.exists("snli_train_sent2.npy") and \
	os.path.exists("snli_train_labels.npy"):
		print ("SNLI preprocessed train model saved")
	else:
		print ("Save error")

else:
	print ("Loading preprocessed corpus...")
	sent1, sent2, y = np.load("snli_train_sent1.npy"), np.load("snli_train_sent2.npy"), np.load("snli_train_labels.npy")
	print ('Shape of sent1 tensor: ', sent1.shape)
	print ('Shape of sent2 tensor: ', sent2.shape)
	print ('Shape of labels tensor: ', y.shape)

"""
Preparing the Embedding Layer (coming soon...)

"""
