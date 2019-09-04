# -*- coding: utf-8 -*-
"""
@author: Sagar
"""
g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

import time
import sys
import numpy as np

# neural network class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        np.random.seed(1)
        self.pre_process_data(reviews, labels)
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):
                review_vocab.add(word)

        # Converting the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        label_vocab = set()
        # populating label_vocab with all of the words in the given labels.
        for label in labels:
            label_vocab.add(label)
        # Converting the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)

        # Storing the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Creating a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # populating self.word2index with indices for all the words in self.review_vocab
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        # Creating a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Storing the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Storing the learning rate
        self.learning_rate = learning_rate

        # Initialize weights between the input layer and the hidden layer
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))

        # Initialize weights between the hidden layer and the output layer
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5,
                                                (self.hidden_nodes, self.output_nodes))

        ## New : Removed self.layer_0; added self.layer_1
        # The input layer, a two-dimensional matrix with shape 1 x hidden_nodes
        self.layer_1 = np.zeros((1,hidden_nodes))

    def get_target_for_label(self,label):
        if label=='NEGATIVE':
            return 0
        else:
            return 1

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_output_2_derivative(self,output):
        return output*(1-output)

    def train(self, training_reviews_raw, training_labels):
         ## New : changed name of first parameter form 'training_reviews'
         #                     to 'training_reviews_raw'
        ##pre-process training reviews so we can deal directly with the indices of non-zero inputs
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))
        # checking we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))

        # Keeping track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remembering the time when we started for printing time statistics
        start = time.time()

        # looping through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # Getting the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]

            # Implementing the forward pass through the network.
            ## New : Removed call to 'update_input_layer' function because 'layer_0' is no longer used

            # Hidden layer
            ## New : Add in only the weights for non-zero items
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]

            # Output layer
            ## New : changed to use 'self.layer_1' instead of 'local layer_1'
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            # Implementing the back propagation pass here.
            ### Backward pass ###

            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label)
            # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error
            # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            ## New : changed to use 'self.layer_1' instead of local 'layer_1'
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate

            ## New : Only update the weights that were used in the forward pass
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate

            # To determine if the prediction was
            # correct, we check that the absolute value of the output error
            # is less than 0.5. If so, add one to the correct_so_far count.
            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            # printing out our prediction accuracy and speed
            # throughout the training process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2400 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        # keeping track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Looping through each of the given reviews and calling run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1

            # printing out the prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")

    def run(self, review):
        ## New: Removed call to update_input_layer function
        #                     because layer_0 is no longer used

        # Hidden layer
        ## New: Identify the indices used in the review and then add
        #                     just those weights to layer_1
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]

        # Output layer
        ## New : changed to use self.layer_1 instead of local layer_1
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        if(layer_2[0] >= 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"

    def testnew(self, newreview):
        pred = self.run(newreview)
        print(pred)

def new():
    newreview = input("enter a review to test: ")
    mlp.testnew(newreview)


mlp = SentimentNetwork(reviews[:24000],labels[:24000], learning_rate=0.1)

mlp.train(reviews[:24000],labels[:24000])

print("\n\ntesting now:")

mlp.test(reviews[24001:25000],labels[24001:25000])

print("\n")

new()



