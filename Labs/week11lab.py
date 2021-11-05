import sys
from IPython.core.ultratb import ColorTB
sys.excepthook = ColorTB()

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint

python_image = plt.imread('/Users/pallavrouth/Dropbox/Teaching/Python bootcamp/Week 11/logo.jpg')
plt.imshow(python_image)
plt.show()

# color images are stored in 3D array
python_image.shape # [height,width,RBG] First 2 spatial dimension - gives us a pixel
# last dimension gives us RBG in each pixel
python_image[220,220] # indexing into the spatial dimension gives us the RGB composition as array

some_pixels = python_image[200:220,200:220]
plt.imshow(some_pixels)
plt.show()

python_image_copy = np.copy(python_image)
python_image_copy[:, :, 0] = 0
python_image_copy[:, :, 1] = 0
plt.imshow(python_image_copy)
plt.show()

# modify specific parts of the image
python_image2 = np.copy(python_image)
python_image2[200:250,200:250,:] = [100,0,0]
plt.imshow(python_image2)
plt.show()

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
train_labels.shape
test_images.shape
test_labels.shape

model1 = Sequential()
model1.add(Dense(10, activation = 'relu', input_shape = (784,))) 
model1.add(Dense(10, activation = 'relu'))
model1.add(Dense(10, activation = 'softmax'))
model1.summary()

model1.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

train_images = train_images.reshape((60000,784))
train_images.shape
len(train_images[1,])
model1.fit(train_images,train_labels,validation_split = 0.2,epochs = 2,verbose = 0)

test_images = test_images.reshape((10000,784))
model1.evaluate(test_images,test_labels)



# CNN

# convolutions as feature detection system
# one edge
img_array = np.array([0,0,0,0,0,1,1,1,1,1])

kernel = np.array([-1,1]) # goes from small (left) to large (right)
# kernel slides over image array
kernel * img_array[0:2] # show math - dot product in 1D
(kernel * img_array[0:2]).sum() # sum the result
# repeat
(kernel * img_array[1:3]).sum() #...
(kernel * img_array[4:6]).sum() 
(kernel * img_array[5:7]).sum() #.. collect these result in colvolution


convolution1 = np.zeros(9)
for ii in range(8):
    convolution1[ii] = (kernel * img_array[ii:ii+2]).sum()
convolution1


# multiple edges
img_array = np.array([0,0,1,1,0,0,1,1,0,0])
kernel = np.array([-1,1])
convolution2 = np.zeros(9)
for ii in range(8):
    convolution2[ii] = (kernel * img_array[ii:ii+2]).sum()
convolution2


# in practice
from keras.datasets import fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
one_img = train_images[2,:,:]
one_img.shape
one_img[11:20,11:20]
plt.imshow(one_img, cmap = 'gray')
plt.show() # save it

kernel = np.array([[-1,1],
                   [-1,1]])
kernel.shape

convolution3 = np.zeros((28,28))
for ii in range(27):
    for jj in range(27):
        window = one_img[ii:ii+2,jj:jj+2]
        convolution3[ii,jj] = np.sum(window * kernel)
convolution3[11:20,11:20]
plt.imshow(convolution3, cmap = 'gray')
plt.show()




# adding convolutional layers
from keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
train_labels.shape
test_images.shape
test_labels.shape

model2 = Sequential()
model2.add(Conv2D(10, kernel_size = 3, activation = 'relu',input_shape = (28, 28, 1)))
model2.add(Flatten()) #  translate between the image processing and classification part of your network
model2.add(Dense(10, activation = 'relu'))
model2.add(Dense(10, activation = 'softmax'))
model2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

train_images = train_images.reshape((-1,28,28,1))
train_images.shape
train_labels.shape

model2.fit(train_images,train_labels,validation_split = 0.2,
           epochs = 2, batch_size = 10)
test_images = test_images.reshape((-1,28,28,1))
model2.evaluate(test_images,test_labels)


from keras.callbacks import ModelCheckpoint
filepath = '/Users/pallavrouth/Dropbox/Teaching/Python bootcamp/Week 11/weights.hdf5'
checkpoint = ModelCheckpoint(filepath = filepath,
                             monitor = 'val_loss', verbose = 1,
                             save_best_only = True)
callbacks_list = [checkpoint]

# now fit
train_images = train_images.reshape((-1,28,28,1))
train_images.shape
train_labels.shape
training = model2.fit(train_images, train_labels, 
                      validation_split = 0.3, epochs = 5, 
                      batch_size = 10, callbacks = callbacks_list,
                      verbose = 0)


history = training.history
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.show()

# Regularization example
# 1. Dropout rate
# model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 # input_shape=(img_rows, img_cols, 1)))

# Add dropout applied to the first layer with 20%.
# 20 pct of units are removed at random
# model.add(Dropout(0.2))
# model.add(Conv2D(5, kernel_size=2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(3, activation='softmax'))

# 2. Batch normalization
# model.add(BatchNormalization())

model2.layers
model2.layers[0]
conv1 = model2.layers[0]
weights_conv1 = conv1.get_weights()
# The first item in this list is an array that holds the values of the 
# weights for the convolutional kernels for this layer. 
kernels = weights_conv1[0]
kernels.shape
# The first 2 dimensions denote the kernel size. 
# The third dimension denotes the number of channels in the kernels. 
# The last dimension denotes the number of kernels in this layer
kernel1 = kernels[:,:,:,0]
kernel1 = kernels[:,:,0,0]
plt.imshow(one_img, cmap = 'gray')
plt.show() # save it

def convolution(image,kernel):
    new_image = np.zeros((28//2,28//2))
    for ii in range(new_image.shape[0]):
        for jj in range(new_image.shape[1]):
            new_image[ii,jj] = np.max(one_img[ii*2:ii*2+2,jj*2:jj*2+2])
    plt.imshow(new_image, cmap = 'gray')
    plt.show()

convolution(one_img,kernel1)


kernel2 = kernels[:,:,0,1]
convolution(one_img,kernel2)

kernel6 = kernels[:,:,0,5]
convolution(one_img,kernel6)

model2.summary()

from keras.layers import MaxPool2D

model3 = Sequential()
model3.add(Conv2D(10, kernel_size = 3, activation = 'relu',input_shape = (28, 28, 1)))
model3.add(MaxPool2D(2))
model3.add(Flatten()) #  translate between the image processing and classification part of your network
model3.add(Dense(10, activation = 'relu'))
model3.add(Dense(10, activation = 'softmax'))
model3.summary()
model3.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
one_img = train_images[5,:,:]
one_img.shape
one_img[11:20,11:20]
plt.imshow(one_img, cmap = 'gray')
plt.show() # save it

new_image = np.zeros((28//2,28//2))
new_image[0,0] = np.max(one_img[0:2,0:2]) # taking a 2 by 2 and coverting to 1 by 1
# repeat ...
new_image[0,1] = np.max(one_img[0:2,2:4])
new_image[1,0] = np.max(one_img[2:4,0:2])

new_image = np.zeros((28//2,28//2))
for ii in range(14):
    for jj in range(14):
        new_image[ii,jj] = np.max(one_img[ii*2:ii*2+2,jj*2:jj*2+2])
plt.imshow(new_image, cmap = 'gray')
plt.show()


#------- Friday

#--- Text preprocessing
# 1. Lower case words
# 2. Remove extra white spaces 
# 3. Removing numbers
# 4. Removing punctuations
# 5. Lemmatize / Stem (optional)
# 6. Remove stopwords

text = "Please, go to the store and      buy a carton of milk and if they have eggs, get 6."
print(text)

# 1. Lower case everything
text_p1 = text.lower()
print(text_p1)

# 2. Remove extra white spaces
import re
text_p2 = re.sub(' +', ' ', text_p1)
print(text_p2)

# 3. remove punctuation
text_p3 = re.sub(r'[^\w\s]', '', text_p2)
print(text_p3)

# 4. remove digits
text_p4 = re.sub(" \d+", '', text_p3)
print(text_p4)

# 5. Lemmatize
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text_p4)

for token in doc:
    print(token.text)

for token in doc:
    print((token.text,token.lemma_))

lemmas = []
for token in doc:
    lemmas.append(token.lemma_)

# stop words
my_stops = ['to','the','and','a','they']

no_stop = []
for lemma in lemmas:
    if lemma not in my_stops:
        no_stop.append(lemma)

# return the original form
text_p5 = ' '.join(no_stop)
print(text_p5)

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
        if lemma not in spacy_stopwords:
            no_stop.append(lemma)
    text_p5 = ' '.join(no_stop)
    return text_p5
    
sentences = ["Count Lev Nikolayevich Tolstoy, usually referred to in English as Leo Tolstoy, was a Russian writer.",
             "He received nominations for the Nobel Prize in Literature every year from 1902 to 1906 and for the Nobel Peace Prize in 1901, 1902, and 1909."]

processed_sentences = []
for i in range(len(sentences)):
    output = preprocess_text(sentences[i])
    processed_sentences.append(output)


# Linguistic features
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Parts of speech
for token in doc:
    print(token.text,token.pos_)

# Lemmas
for token in doc:
    print(token.text,token.lemma_)

# Dependencies
for token in doc:
    print(token.text,token.dep_)


import json

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


sentences_processed = []
for i in range(len(sentences)):
    output = preprocess_text(sentences[i])
    sentences_processed.append(output)
