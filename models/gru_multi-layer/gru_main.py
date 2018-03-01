import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GRU, Embedding, Dropout, Activation, LSTM
from keras.layers.merge import add, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D, BatchNormalization
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import ModelCheckpoint

path = '../../input/'
TRAIN_DATA_FILE=f'{path}train.csv'
TEST_DATA_FILE=f'{path}test.csv'
EMBEDDING_FILE=f'{path}glove.6B.50d.txt'

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

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf8"))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print('Retrieving model')
inp = Input(shape=(maxlen,))

one = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
one = Bidirectional(LSTM(8, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l1(0.0001)))(one)
one = BatchNormalization(momentum=0.99, epsilon=0.001)(one)
one = GlobalMaxPool1D()(one)

two = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
two = Bidirectional(GRU(8, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l1(0.0001)))(two)
two = BatchNormalization(momentum=0.99, epsilon=0.001)(two)
two = GlobalMaxPool1D()(two)

#three = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#three = Bidirectional(LSTM(4, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l1(0.0001)))(three)
#three = BatchNormalization(momentum=0.99, epsilon=0.001)(three)
#three = GlobalMaxPool1D()(three)

#four = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#four = Bidirectional(GRU(2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l1(0.0001)))(four)
#four = BatchNormalization(momentum=0.99, epsilon=0.001)(four)
#four = GlobalMaxPool1D()(four)

x = concatenate([one, two])

x = Dense(16, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(6, activation="softmax")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

print('Training')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='auto')
model.fit(X_t, y, batch_size=16, epochs=2)

print('Predicting')
y_test = model.predict([X_te], batch_size=1024, verbose=1)

print('Retrieving sample submissions')
sample_submission = pd.read_csv(f'{path}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('../../submissions/gru_2layer_embedding_multi_roc.csv', index=False)