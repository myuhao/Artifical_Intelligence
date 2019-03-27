# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

import math

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        #self.lambda_mixture = 0.86752
        self.lambda_mixture = 0.86
        self.label_dict = {}
        self.num_class = 14
        self.prior = []
        self.vocab = 0
        self.class_vocab = []
        self.vocab_bigram = 0
        self.class_vocab_bigram = []
        self.label_dict_bigram = {}

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
            label is from 1 to 14
        """

        # TODO: Write your code here
        k_smooth = 4

        for j in range(self.num_class):
            self.prior.append(0)
            self.class_vocab.append(0)
            self.class_vocab_bigram.append(0)

        for i in range(len(train_label)):
            text = train_set[i]
            label = train_label[i]
            self.prior[label-1] += k_smooth
            for word in text:
                try:
                    self.label_dict[word]
                except KeyError:
                    self.label_dict[word] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # 14 classes

                self.label_dict[word][label-1] += 1  # label form 1-14 so -1 as index

            for k in range(len(text)):
                if k < len(text) - 1:
                    bigram = text[k] + "+" + text[k+1]
                    try:
                        self.label_dict_bigram[bigram]
                    except KeyError:
                        self.label_dict_bigram[bigram] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # 14 classes

                    self.label_dict_bigram[bigram][label-1] += 1


        #print(self.label_dict_bigram)
        #print(len(self.label_dict_bigram))
        self.vocab = len(self.label_dict)

        for l in range(len(self.prior)):
            self.prior[l] /= len(train_label)
            self.prior[l] = math.log10(self.prior[l])

        print(self.prior)
        # unigram likelihood
        for key in self.label_dict:
            #print(key)
            #print(self.label_dict[key])
            for i in range(len(self.label_dict[key])):
                self.class_vocab[i] += self.label_dict[key][i]
                self.label_dict[key][i] += 1

        for key in self.label_dict:
            for i in range(len(self.label_dict[key])):
                self.label_dict[key][i] /= (self.vocab + self.class_vocab[i]*k_smooth)
                self.label_dict[key][i] = math.log10(self.label_dict[key][i])

        # bigram likelihood
        for key in self.label_dict_bigram:
            for i in range(len(self.label_dict_bigram[key])):
                self.class_vocab_bigram[i] += self.label_dict_bigram[key][i]
                self.label_dict_bigram[key][i] += 1

        for key in self.label_dict_bigram:
            for i in range(len(self.label_dict_bigram[key])):
                self.label_dict_bigram[key][i] /= (self.vocab + self.class_vocab_bigram[i]*k_smooth)
                self.label_dict_bigram[key][i] = math.log10(self.label_dict_bigram[key][i])

       # print(self.label_dict)


    def predict(self, x_set, dev_label,lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """

        accuracy = 0.0
        result = []
        correctness = []
        lambda_mix = self.lambda_mixture

        # TODO: Write your code here
        for i in range(len(dev_label)):
            test_text = x_set[i]
            test_label = dev_label[i]
            posterior = list.copy(self.prior)
            # unigram contribution
            for word in test_text:
                try:
                    likelihood = self.label_dict[word]
                    for j in range(len(posterior)):
                        posterior[j] += likelihood[j]*(1-lambda_mix)

                except KeyError:
                    # do nothing for unseen words
                    pass
                #print(posterior)

            # bigram contribution

            for t in range(len(test_text)):
                if t < len(test_text) - 1:
                    test_bigram = test_text[t] + "+" + test_text[t+1]
                    try:
                        likelihood = self.label_dict_bigram[test_bigram]
                        for p in range(len(posterior)):
                            posterior[p] += likelihood[p] * lambda_mix
                    except KeyError:
                        pass


            max = -float("inf")
            #print(posterior)
            for k in range(len(posterior)):
                if posterior[k] > max:
                    max = posterior[k]
                    argmax = k
            result.append(argmax+1)
            if argmax == (test_label-1):
                correctness.append(1)
                accuracy += 1
            else:
                correctness.append(0)

        accuracy /= len(dev_label)

        print(correctness)
        #print(dev_label)

        return accuracy,result

