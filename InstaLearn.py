import re
from bert_serving.client import BertClient
import scipy.spatial as sp
from termcolor import colored
import sys
import numpy as np


def distance_matrix(matrix1, matrix2):
    """
    Calculate the distance matrix between 2 embedding matrix.

    Args:
        matrix1: (n1, d) NumPy array (a sequence embedding with n1 tokens, represented by a d hidden units BERT model)
        matrix2: (n2, d) NumPy array (a sequence embedding with n2 tokens, represented by a d hidden units BERT model)
    returns:
        (n1, n2) Numpy array (indicates token wise cosine distance between these 2 sequence embedding.)
    """
    return 1 - sp.distance.cdist(matrix1, matrix2, 'cosine')


def tokenize(sentence):
    """
    Naive tokenize (punctuation splitting, white space, etc.)
    Should use word-piece tokenization in the future

    Args:
        sentence: string
    returns:
        list(string)
    """
    return re.sub('(?<! )(?=[.,!?()"\'])|(?<=[.,?()"\'])(?! )', r' ', sentence).split()


def print_sen(sentence_tokenized, label_dict):
    """
    Visualize the sentence, using the Red color to identify the * prefix word, and use the blue color to identify the
    ! prefix word.

    Args:
        sentence_tokenized: list(string)
        label_dict: dict (set) (a dictionary which contain the indexes of the target words.)
    returns:
        None
    """
    length = 0
    for i, w in enumerate(sentence_tokenized):
        if length + len(w) > 130:
            # Make sure not to print too long in one single line.
            length = 0
            sys.stdout.write('\n')
        if length and w[0].isalnum() and sentence_tokenized[i - 1] not in "-'(\"":
            # add white space if w is not punctuation and not the start of a line
            w = ' ' + w
        if i + 1 in label_dict['*']:
            sys.stdout.write(colored(w, 'red'))
        elif i + 1 in label_dict['!']:
            sys.stdout.write(colored(w, 'blue'))
        else:
            sys.stdout.write(w)
        length += len(w)

    sys.stdout.write('\n')
    sys.stdout.flush()


class InstaLearn(object):
    def __init__(self, ip='localhost', port=5555, port_out=5556):
        self.bertclient = BertClient(ip=ip, port=port, port_out=port_out)
        self.train_data = None
        self.target_inx = {'*': set(), '!': set()}
        self.train_embedding = None
        self.threshold = {'*': 0.9, '!': 0.9}

    def train(self, sentence):
        """
        Given a sentence, tokenize it, send it to the bert-service, get back the sequence
        embedding representation of it, then calculate the inference threshold, save the
        result in the class, at last visualize the sentence.
        Args:
            sentence: string

        returns: None
        """
        print('\nStart training.......................')
        self.train_data = tokenize(sentence)
        for i, s in enumerate(self.train_data):
            if s[0] in '*!' and len(s) > 1:
                # Index should +1 because the BERT will add [CLS] at the beginning of the sentence
                self.target_inx[s[0]].add(i + 1)
                self.train_data[i] = s[1:]
        vec = self.bertclient.encode([[s.lower() for s in self.train_data]], is_tokenized=True)
        # Index length should +2 because the BERT will add [CLS] and [SEP] at
        # the beginning and end of the sentence
        self.train_embedding = vec[0][:len(self.train_data) + 2]
        # Calculate the inference threshold.
        self.set_threshold()
        # Visualize the training data
        print_sen(self.train_data, self.target_inx)

    def set_threshold(self):
        """
        Calculate and set the inference threshold. Basic idea is for every target word, first find the minimum cosine
        distance between it and other target words, then find the maximum cosine distance between it and other
        non-target words, at last average these 2 values.

        Args:
            None

        returns:
            None
        """
        dist_matrix = distance_matrix(self.train_embedding, self.train_embedding)
        for prefix in self.target_inx:
            target_distances = []
            nontarget_distances = []
            for tinx in self.target_inx[prefix]:
                for i in range(len(self.train_embedding)):
                    if i in self.target_inx[prefix]:
                        # Distances between target words
                        target_distances.append(dist_matrix[tinx][i])
                    else:
                        # Distances between target words and non-target words
                        nontarget_distances.append(dist_matrix[tinx][i])
            # calculate threshold: (min target word distance + max non-target word distance)/2
            self.threshold[prefix] = (min(0.8, max(nontarget_distances)) + min(0.9, min(target_distances))) / 2

    def inference(self, sentence):
        """
        Given a sentence, tokenize it, send it to the bert-service, get back the sequence
        embedding representation of it, then calculate the distance matrix between the train-embedding
        and the inference-embedding, find the distances that is bigger than the threshold, visualize and
        return the result

        Args:
            sentence: string

        returns:
            tokenized: list (string) (tokenized inference sentence)
            inf_label: dict (set) (a dictionary which record the index of the words that are similar to the
            target words in training sentence)
        """
        assert self.train_data is not None
        print('\nInference.......................')
        tokenized = tokenize(sentence)
        inf_embedding = self.bertclient.encode([[s.lower() for s in tokenized]], is_tokenized=True)[0][
                        :len(tokenized) + 2]
        inf_label = {"*": set(), "!": set()}

        dist_matrix = distance_matrix(inf_embedding, self.train_embedding)
        for i, word_distance in enumerate(dist_matrix):
            nearest = np.argmax(word_distance)
            if nearest in self.target_inx['*'] and word_distance[nearest] > self.threshold['*']:
                inf_label['*'].add(i)
            if nearest in self.target_inx['!'] and word_distance[nearest] > self.threshold['!']:
                inf_label['!'].add(i)

        print_sen(tokenized, inf_label)
        return tokenized, inf_label


if __name__ == "__main__":
    il = InstaLearn()
    while True:
        query = input('\ninput training data: ')
        il.train(query)
        query = input('\ninput inference data: ')
        _, _ = il.inference(query)
