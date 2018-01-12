import tensorflow as tf
from tensorflow.contrib import layers
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import math
from random import randint
import pickle
import os

# This Word2Vec implementation is largely based on this paper
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
# It's a bit old, but Word2Vec is still SOTA and relatively simple, so I'm going with it

# Check out Tensorflow's documentation which is pretty good for Word2Vec
# https://www.tensorflow.org/tutorials/word2vec

wordVecDimensions = 100
batchSize = 128
numNegativeSample = 100
windowSize = 7
numIterations = 1000000

# This function just takes in the conversation data and makes it 
# into one huge string, and then uses a Counter to identify words
# and the number of occurences
def processDataset(filename):
    openedFile = open(filename, 'r', encoding='utf8')
    allLines = openedFile.readlines()
    myStr = ""
    for line in allLines:
        myStr += line
    finalDict = Counter(myStr.split())
    return myStr, finalDict

def createTrainingMatrices(dictionary, corpus):
    allUniqueWords = dictionary.keys()
    allWords = corpus.split()
    numTotalWords = len(allWords)
    X=[]
    y=[]
    for i in range(numTotalWords):
        if i % 1000 == 0:
            print('Finished %d/%d total words' % (i, numTotalWords))
        wordsAfter = allWords[i + 1:i + windowSize + 1]
        wordsBefore = allWords[max(0, i - windowSize):i]
        wordsAdded = wordsAfter + wordsBefore
        for word in wordsAdded:
            X.append(list(allUniqueWords).index(allWords[i]))
            y.append(list(allUniqueWords).index(word))
    return allUniqueWords, X, y


def getTrainingBatch():
    num = randint(0, numTrainingExamples - batchSize - 1)
    arr = X_train[num:num + batchSize]
    labels = y_train[num:num + batchSize]
    return arr, labels

# Loading the data structures if they are present in the directory
if (os.path.isfile('Word2VecXTrain.npy') and os.path.isfile('Word2VecYTrain.npy') and os.path.isfile('wordList.txt')):
    X = np.load('Word2VecXTrain.npy')
    y = np.load('Word2VecYTrain.npy')
    y = np.reshape(y, [-1, 1])
    print('Finished loading training matrices')
    with open("wordList.txt", "rb") as fp:
        wordList = pickle.load(fp)
    print('Finished loading word list')

else:
    fullCorpus, datasetDictionary = processDataset('conversationData.txt')
    print('Finished parsing and cleaning dataset')
    print('Full corpus length: %d' % len(fullCorpus))
    print('Dictionary length: %d' % len(datasetDictionary))
    wordList, X, y = createTrainingMatrices(datasetDictionary, fullCorpus)
    y = np.reshape(y, [-1, 1])
    print('Finished creating training matrices')
    np.save('Word2VecXTrain.npy', X)
    np.save('Word2VecYTrain.npy', y)
    with open("wordList.txt", "wb") as fp:
        pickle.dump(list(wordList), fp)

numTrainingExamples = len(X)
vocabSize = len(wordList)

graph = tf.Graph()

# Create tensor variables, network layers, labels, and loss using Tensorflow
with graph.as_default():
    # Input data.
    inputs = tf.placeholder(tf.int32, shape=[None])
    labels = tf.placeholder(tf.int32, shape=[None, 1])

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    # Look up embeddings for inputs. tf.random_uniform([vocabSize,wordVecDimensions],-1.0, 1.0)
    embeddings = tf.get_variable("E", shape=[vocabSize, wordVecDimensions],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 regularizer=regularizer)
    embed = tf.nn.embedding_lookup(embeddings, inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.get_variable("W", shape=[vocabSize, wordVecDimensions],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  regularizer=regularizer)

    nce_biases = tf.Variable(tf.zeros([vocabSize]))


    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=labels,
                                         inputs=embed,
                                         num_sampled=batchSize/2,
                                         num_classes=vocabSize)
                          )

    # Decaying the learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.05
    learning_rate = tf.reduce_max([tf.train.exponential_decay(starter_learning_rate, global_step, 1500, 0.96,
                                                              staircase=True),
                                   0.00001])

    # Construct the AdamOptimizer with default parameters
    optimizer = tf.train.AdamOptimizer(beta1=0.9,
                                       beta2=0.99,
                                       epsilon=1e-6,
                                       learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # Add variable initializer.
    init = tf.global_variables_initializer()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

print("y stats: ", max(y_train), min(y_train))
print("X stats: ", max(X_train), min(X_train))

training_loss = []
val_loss = []
embeddings_mat = None
cum_training_loss = 0
min_val_loss = 100000
print_step = numIterations * 0.25 / 100

with tf.Session(graph=graph) as session:
    session.run(init)

    for i in range(numIterations):
        trainInputs, trainLabels = getTrainingBatch()
        _, curLoss = session.run([optimizer, loss], feed_dict={inputs: trainInputs, labels: trainLabels})

        if not np.isnan(curLoss):
            cum_training_loss += curLoss

        if i % print_step == 0:
            print("Iteration: ", i, " Learning rate: ", session.run(learning_rate))

            if i > 0:
                cum_training_loss /= print_step

            training_loss.append(cum_training_loss)
            print('Training loss: ', cum_training_loss)
            cum_training_loss = 0

            val_step_loss = session.run(loss, feed_dict={inputs: X_val, labels: y_val})
            val_loss.append(val_step_loss)
            print('Validation loss: ', val_step_loss)

            if val_step_loss < min_val_loss:
                min_val_loss = val_step_loss
                embeddings_mat = embeddings.eval()

print("Lower validation loss: ", min_val_loss)
print('Saving the word embedding matrix')
np.save('embeddingMatrix.npy', embeddings_mat)
