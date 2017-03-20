import pandas as pd
import os
import numpy as np
import keras
import keras.backend as K 
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, TimeDistributed, Dropout, Input, Dense, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.recurrent import SimpleRNN, LSTM

def load_corpus(path, usecols=['gold_label', 'sentence1_binary_parse', 'sentence2_binary_parse']):
	df = pd.read_csv(path,
		delimiter='\t', usecols=usecols)
	return df

def drop_unclassified(df):
    #Drop samples without gold label
    df = df[df.gold_label != '-']
    return df

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

labels_dict = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

def classes_to_categories(df):
	df.loc[df['gold_label'] == 'entailment', 'gold_label'] = labels_dict['entailment']
	df.loc[df['gold_label'] == 'contradiction', 'gold_label'] = labels_dict['contradiction']
	df.loc[df['gold_label'] == 'neutral', 'gold_label'] = labels_dict['neutral']

def categories_to_categorical(labels, num_classes):
	return to_categorical(labels, num_classes=num_classes)

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
8. Pad sequences, so they all have the same length (default: 42)

"""

print ("Preprocessing train corpus...")
path_train = 'snli_1.0/snli_1.0_train.txt'
df_train = load_corpus(path_train)
df_train = drop_unclassified(df_train)
classes_to_categories(df_train)
df_train['sentence1_binary_parse'] = df_train['sentence1_binary_parse'].apply(
	lambda x: extract_tokens_from_binary_parse(str(x)))
df_train['sentence2_binary_parse'] = df_train['sentence2_binary_parse'].apply(
	lambda x: extract_tokens_from_binary_parse(str(x)))

MAX_LEN = 42

tokenizer = Tokenizer(lower=False, filters='')
sent1_train, sent2_train = tokens_to_sentences(df_train['sentence1_binary_parse'], 
	df_train['sentence2_binary_parse'])
tokenizer.fit_on_texts(sent1_train + sent2_train)

y_train = categories_to_categorical(df_train['gold_label'], 3)

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
sentences_to_padded_index_sequences = lambda data: to_seq(data)

sent1_train, sent2_train = sentences_to_padded_index_sequences(sent1_train), \
sentences_to_padded_index_sequences(sent2_train)
print ('Shape of sent1_train tensor: ' + str(sent1_train.shape))
print ('Shape of sent2_train tensor: ' + str(sent2_train.shape))
print ('Shape of label_train tensor: ' + str(y_train.shape))

print ("Preprocessing dev corpus...")
path_dev = 'snli_1.0/snli_1.0_dev.txt'
df_dev = load_corpus(path_dev)
df_dev = drop_unclassified(df_dev)
classes_to_categories(df_dev)
df_dev['sentence1_binary_parse'] = df_dev['sentence1_binary_parse'].apply(
	lambda x: extract_tokens_from_binary_parse(str(x)))
df_dev['sentence2_binary_parse'] = df_dev['sentence2_binary_parse'].apply(
	lambda x: extract_tokens_from_binary_parse(str(x)))

sent1_dev, sent2_dev = tokens_to_sentences(df_dev['sentence1_binary_parse'], 
	df_dev['sentence2_binary_parse'])

y_dev = categories_to_categorical(df_dev['gold_label'], 3)

sent1_dev, sent2_dev = sentences_to_padded_index_sequences(sent1_dev), \
sentences_to_padded_index_sequences(sent2_dev)

print ('Shape of sent1_dev tensor: ' + str(sent1_dev.shape))
print ('Shape of sent2_dev tensor: '+ str(sent2_dev.shape))
print ('Shape of label_dev tensor: '+ str(y_dev.shape))

print ("Preprocessing test corpus...")
path_test = 'snli_1.0/snli_1.0_test.txt'
df_test = load_corpus(path_test)
df_test = drop_unclassified(df_test)
classes_to_categories(df_test)
df_test['sentence1_binary_parse'] = df_test['sentence1_binary_parse'].apply(
	lambda x: extract_tokens_from_binary_parse(str(x)))
df_test['sentence2_binary_parse'] = df_test['sentence2_binary_parse'].apply(
	lambda x: extract_tokens_from_binary_parse(str(x)))

sent1_test, sent2_test = tokens_to_sentences(df_test['sentence1_binary_parse'], 
	df_test['sentence2_binary_parse'])

y_test = categories_to_categorical(df_test['gold_label'], 3)

sent1_test, sent2_test = sentences_to_padded_index_sequences(sent1_test), \
sentences_to_padded_index_sequences(sent2_test)

print ('Shape of sent1_test tensor: ' + str(sent1_test.shape))
print ('Shape of sent2_test tensor: '+ str(sent2_test.shape))
print ('Shape of label_test tensor: '+ str(y_test.shape))

"""
Preparing the Embedding Layer

"""
print ("Preparing Embedding Layer...")
GLOVE_DIR = ''
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs

print ('Found %s word vectors.' % len(embeddings_index))

"""
Baseline model

"""
print ("Building model...")
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(tokenizer.word_index) + 1,
				100,
				weights=[embedding_matrix],
				input_length=MAX_LEN,
				trainable=False)

sum_embeddings_layer = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(100, ))
rnn_layer = SimpleRNN(units=100, activation='tanh', dropout=0.5,
			recurrent_dropout=0.5, implementation=0)
lstm_layer = LSTM(units=100, activation='sigmoid', recurrent_activation='tanh',
			dropout=0.5, recurrent_dropout=0.5, implementation=0)
translate_layer = Dense(units=100, activation='relu')

premise = Input(shape=(MAX_LEN, ), dtype='int32')
hypothesis = Input(shape=(MAX_LEN, ), dtype='int32')
prem = embedding_layer(premise)
hypo = embedding_layer(hypothesis)
prem = sum_embeddings_layer(prem)
hypo = sum_embeddings_layer(hypo)
prem = translate_layer(prem)
hypo = translate_layer(hypo)
prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)
merged_sentences = concatenate([prem, hypo], axis=1)
x = Dense(200, activation='relu', kernel_regularizer=l2(1e-2))(merged_sentences)
x = BatchNormalization()(x)
x = Dense(200, activation='relu', kernel_regularizer=l2(1e-2))(x)
x = BatchNormalization()(x)
x = Dense(200, activation='relu', kernel_regularizer=l2(1e-2))(x)
x = BatchNormalization()(x)
pred = Dense(3, activation='softmax')(x)

model = Model(outputs=pred, inputs=[premise, hypothesis])

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_acc', patience=4),\
			TensorBoard(log_dir='./logs/snli/run1', histogram_freq=1, write_graph=False),\
			ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=5)]
model.fit([sent1_train, sent2_train], y_train, batch_size=512, epochs=100, 
	validation_data=([sent1_dev, sent2_dev], y_dev), callbacks=callbacks)
loss, acc = model.evaluate([sent1_test, sent2_test], y_test)

print ('\nloss: ' + str(loss) + '  acc: ' + str(acc))
