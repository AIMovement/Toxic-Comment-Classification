import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GRU, Embedding, Dropout, Activation
from keras.layers.merge import add, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D, BatchNormalization
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers

path = '../../input/'
TRAIN_DATA_FILE=f'{path}train.csv'
TEST_DATA_FILE=f'{path}test.csv'

embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

print('Load train data')
train = pd.read_csv(TRAIN_DATA_FILE)

print('Load test data')
test = pd.read_csv(TEST_DATA_FILE)

print('Remove NaNs')
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

print('Tokenizing')
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

print('Padding')
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

print('Retrieving model')
inp = Input(shape=(maxlen,))

left = Embedding(max_features, embed_size)(inp)
left = Bidirectional(GRU(16, return_sequences=True, dropout=0.1, recurrent_dropout=0.2, kernel_regularizer=regularizers.l1(0.0001)))(left)
left = BatchNormalization(momentum=0.99, epsilon=0.001)(left)
left = GlobalMaxPool1D()(left)

right = Embedding(max_features, embed_size)(inp)
right = Bidirectional(GRU(16, return_sequences=True, dropout=0.2, recurrent_dropout=0.1, kernel_regularizer=regularizers.l1(0.0001)))(right)
right = BatchNormalization(momentum=0.99, epsilon=0.001)(right)
right = GlobalMaxPool1D()(right)

x = concatenate([left, right])

x = Dense(16, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(6, activation="softmax")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

print('Training')
model.fit(X_t, y, batch_size=16, epochs=2)

print('Predicting')
y_test = model.predict([X_te], batch_size=1024, verbose=1)

print('Retrieving sample submissions')
sample_submission = pd.read_csv(f'{path}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('../../submissions/gru_multi3_batchnorm.csv', index=False)