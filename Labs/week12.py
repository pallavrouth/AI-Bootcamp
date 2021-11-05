import sys
from IPython.core.ultratb import ColorTB
sys.excepthook = ColorTB()

#-------------- Deep learning for NLP -----------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sentences = ['I love my cat', 'I love learning my cat and AI']

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
tokenizer.texts_to_sequences(sentences)

test_sentence = ['I like my watch','I like my computer']
tokenizer = Tokenizer(num_words = 100, oov_token = 'new_token')
tokenizer.texts_to_sequences(test_sentence)
tokenizer.fit_on_texts(test_sentence)
word_index = tokenizer.word_index
print(word_index)


sentences = ['I love my cat', 'AI is interesting but very hard', 'The time now is 8:40 pm']
tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
np.vstack(sequences)

padded_sequences = pad_sequences(sequences)
padded_sequences = pad_sequences(sequences, maxlen = 4)
padded_sequences = pad_sequences(sequences, padding = 'post')
padded_sequences

# creating lists for sentences,labels and urls
sentences = [] # headlines
labels = [] # labels
urls = []
# iterating through the json data and loding 
# the requisite values into our python lists
for line in open("/Users/pallavrouth/Dropbox/Teaching/Python bootcamp/Week 11/sarcasmv2.json",'r'):
    sentences.append(json.loads(line)['headline'])
    labels.append(json.loads(line)['is_sarcastic'])
    urls.append(json.loads(line)['article_link'])


nlp = spacy.load('en_core_web_sm')
def preprocess_text(text):
    text_p1 = text.lower()
    text_p2 = re.sub(' +', ' ', text_p1)
    text_p3 = re.sub(r'[^\w\s]', '', text_p2)
    text_p4 = re.sub(" \d+", '', text_p3)
    doc = nlp(text_p4)
    lemmas = []
    for token in doc:
        lemmas.append(token.lemma_)
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    no_stop = []
    for lemma in lemmas:
        if lemma not in spacy_stopwords and lemma.isalpha:
            no_stop.append(lemma)
    text_p5 = ' '.join(no_stop)
    return text_p5

sentences_processed = []
for i in range(len(sentences)):
    print(i)
    output = preprocess_text(sentences[i])
    sentences_processed.append(output)


df = pd.DataFrame({"labels" : labels, "sentences" : sentences_processed})
df.head()

train_data, test_data = train_test_split(df, test_size = 0.3)
train_data.shape
test_data.shape

vocab_size = 10000
max_length = 100



trunc_type = 'post'
padding_type = 'post'


oov_tok = "<OOV>"
training_size = 20000

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_data['sentences'])
training_sequences = tokenizer.texts_to_sequences(train_data['sentences'])

training_padded = pad_sequences(training_sequences,
                                maxlen = max_length,
                                padding = padding_type,
                                truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test_data['sentences'])
testing_padded = pad_sequences(testing_sequences,
                                maxlen = max_length,
                                padding = padding_type,
                                truncating = trunc_type)

training_labels = np.array(train_data['labels'])
testing_labels = np.array(test_data['labels'])

embedding_dim = 16

# creating a model for sentiment analysis
model  = tf.keras.Sequential([
                # we define an Embedding layer with a vocabulary of 10,000 
                # (e.g. integer encoded words from 0 to 9999, inclusive), 
                # a vector space of 16 dimensions in which each words will be embedded, 
                # defines the size of the output vectors from this layer for each word.
                # and input documents that have max of 100 words each.
                tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
                # Global Average pooling is similar to adding up vectors in this case
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(24, activation = 'relu'),
                tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
num_epochs = 5
history = model.fit(training_padded,training_labels, epochs = num_epochs,validation_split = 0.2, batch_size = 10)
model.evaluate(testing_padded,testing_labels)

#----------------

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Create CountVectorizer object
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
vectorizer = CountVectorizer(max_features = 200) #tfidf vectorizer
# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(df['sentences'])
bow_matrix_array = bow_matrix.toarray()
target = df['labels'].values


new_df = pd.DataFrame(bow_matrix_array, columns = vectorizer.get_feature_names())


features_train, features_test, target_train, target_test = train_test_split(bow_matrix_array, target, test_size = 0.20, random_state = 42)

# build and predict a regression random forest
regr = RandomForestClassifier(max_features = 10, random_state = 0, n_estimators = 50)
mod = regr.fit(features_train, target_train)
target_pred = mod.predict(features_test)
pd.DataFrame(confusion_matrix(target_test, target_pred),columns = ["Pred0", "Pred1"], index = ["Actual0", "Actual1"])


#---- Friday

import pandas as pd
import numpy as np

#data = pd.read_csv('bexar2020.csv', low_memory = False)

sample_data = {'age' : ['lessthan10','10to20','20to39','40andup']}
df = pd.DataFrame(sample_data)

def get_new_age(x):
    if x == 'lessthan10' or x == '10to20':
        return 'lessthan20'
    else:
        return x

df['age2'] = df['age'].apply(get_new_age)
df['age3'] = np.where(df['age'].isin(['20to39','40andup']),df['age'],'lessthan20')

df2 = pd.get_dummies(df,columns = ['age2'], drop_first = True)
df3 = pd.get_dummies(df,columns = ['age2'], drop_first = False)

#loan_data2 = pd.get_dummies(loan_data, columns = ['age'], drop_first=True)
#loan_data['age2'] = np.where(loan_data['age'] <= 2, 1, 0)