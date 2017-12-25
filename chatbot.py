"""
    This is a Retrieval model based chatbot which uses the concept of text classification implemented by building
    our own Artificial Neural Network without Tensorflow.

"""

import nltk
import os
import json
import datetime
import numpy as np
import time
import random
from nltk.stem.lancaster import LancasterStemmer


# Creating a stemmer object.
stemmer = LancasterStemmer()

classes, words, documents = [], [], []
ignore_words = ['?', ',', '.']
training, output, training_data = [], [], []


# Loading training data.
with open('data.json') as data:
    training_data = json.load(data)['data']


# Normalizing using sigmoid function.
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# Convert sigmoid output to its derivative.
def sigmoid_to_derivative(output):
    return output * (1 - output)


# Train the Artificial Neural Network.
def train(X, y, hidden_neurons=30, alpha=0.01, epochs=1000000):
    print ("Training with {0} neurons, alpha: {1}".format(hidden_neurons, str(alpha)))
    print ("Input matrix: {}x{}    Output matrix: {}x{}".format(len(X),len(X[0]),1, len(classes)))

    # Write number of sentences and words.
    record = {
                "sentences": len(X),
                "words": len(X[0])
             }
    with open('log.json', 'w') as f:
        json.dump(record, f, indent=4)
    print(len(documents), 'documents')
    print(len(classes), 'classes')
    print(len(words), 'words')

    np.random.seed(1)

    last_mean_error = 1
    # Randomly initialize our weights with mean 0
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("Delta after "+str(j)+" iterations: " + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("Break: ", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break

        layer_2_delta = layer_2_error * sigmoid_to_derivative(layer_2)

        layer_1_error = layer_2_delta.dot(synapse_1.T)

        layer_1_delta = layer_1_error * sigmoid_to_derivative(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # Store synaptic weights for prediction.
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("Saved synapses to:", synapse_file)


def prepare_data():
    global words, classes, documents, ignore_words, training_data

    # Tokenize the sentences in each class.
    for pattern in training_data:
        for sentence in pattern['sentences']:
            w = nltk.word_tokenize(sentence)
            words.extend(w)
            documents.append((w, pattern['class']))
            if pattern['class'] not in classes:
                classes.append(pattern['class'])

    words = list(set([ stemmer.stem(w.lower()) for w in words if w not in ignore_words ]))

    # Creating training data using one hot encoding method.
    # Creating bag of words for each sentence.
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [ stemmer.stem(w.lower()) for w in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        training.append(bag)
        output_row = [0] * len(classes)
        output_row[classes.index(doc[1])] = 1
        output.append(output_row)

    #print(documents[6][0], ' --> ', training[6])
    #print(documents[6][1], ' --> ', output[6])
    return training, output


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if s == w:
                bag[i] = 1
    return np.array(bag)


def predict(sentence):
    # Loading the calculated synapse values.
    synapse_file = 'synapses.json'
    with open(synapse_file) as data_file:
        synapse = json.load(data_file)
        synapse_0 = np.asarray(synapse['synapse0'])
        synapse_1 = np.asarray(synapse['synapse1'])
        classes = np.asarray(synapse['classes'])
        words = np.asarray(synapse['words'])
    x = bag_of_words(sentence, words)

    # Defining layers for prediction.
    l0 = x
    l1 = sigmoid(np.dot(l0, synapse_0))
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2, classes


def classify(sentence):
    ERROR_THRESHOLD = 0.25
    results, classes = predict(sentence)

    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    #print("Sentence: {}, Classification: {}".format(sentence, return_results))
    return return_results


def change_in_data(_sentences, _words):
    try:
        with open('log.json', 'r') as f:
            log = json.load(f)
        if _sentences != int(log["sentences"]) or _words != int(log["words"]):
            return True
        return False
    except FileNotFoundError:
        return True


if __name__ == '__main__':

    # Preparing data.
    X, y = prepare_data()
    X = np.array(training)
    y = np.array(output)

    if(change_in_data(len(X), len(X[0]))):
        # Training the ANN.
        start_time = time.time()
        train(X, y, hidden_neurons=40, alpha=0.1, epochs=1000000)
        elapsed_time = time.time() - start_time
        print ("Training time:", elapsed_time, "seconds")

    while True:
        # Classifying new sentence.
        _input = input('You: ')
        class_ = classify(_input)[0][0]

        for _class in training_data:
            if _class["class"] == class_:
                print("Bot: " + random.choice(_class["responses"]))
                if _class["class"].strip() == 'goodbye':
                    break
