import re
from bert_serving.client import BertClient
import scipy.spatial as sp
from termcolor import colored
import sys
import numpy as np


class InstaLearn(object):
    def __init__(self, ip='localhost', port=5555, port_out=5556):
        # self.model_dir = model_dir
        self.bertclient = BertClient(ip=ip, port=port, port_out=port_out)
        self.train_data = None
        self.target_inx = {'*': set(), '!': set()}
        self.train_embedding = None
        self.threshold = {'*': 0.9, '!': 0.9}

    def train(self, sentence):
        # Tokenize (punctuation splitting, white space, etc.)
        self.train_data = re.sub('(?<! )(?=[.,!?()"\'])|(?<=[.,?()"\'])(?! )', r' ', sentence).split()
        for i, s in enumerate(self.train_data):
            if s[0] in '*!' and len(s) > 1:
                # Index should +1 because the BERT will add [CLS] at the beginning of the sentence
                self.target_inx[s[0]].add(i + 1)
                self.train_data[i] = s[1:]
        # vec = self.bertclient.encode([self.train_data], is_tokenized=True)
        vec = self.bertclient.encode([[s.lower() for s in self.train_data]], is_tokenized=True)
        # Index length should +2 because the BERT will add [CLS] and [SEP] at
        # the beginning and end of the sentence
        self.train_embedding = vec[0][:len(self.train_data) + 2]
        self.set_threshold()
        print('\nStart training.......................')
        self.print_sen(self.train_data, self.target_inx)

    def set_threshold(self):
        distance_matrix = 1 - sp.distance.cdist(self.train_embedding, self.train_embedding, 'cosine')
        for prefix in self.target_inx:
            target_distances = []
            nontarget_distances = []
            for tinx in self.target_inx[prefix]:

                for i in range(len(self.train_embedding)):
                    if i in self.target_inx[prefix]:
                        # Distances between target words
                        target_distances.append(distance_matrix[tinx][i])
                    else:
                        # Distances between target words and non-target words
                        nontarget_distances.append(distance_matrix[tinx][i])
                # calculate threshold: (min target word distance + max non-target word distance)/2
            self.threshold[prefix] = (min(0.8, max(nontarget_distances)) + min(0.9, min(target_distances))) / 2

    def inference(self, sentence):
        tokenized = re.sub('(?<! )(?=[.,!?()"\'])|(?<=[.,?()"\'])(?! )', r' ', sentence).split()
        inf_embedding = self.bertclient.encode([[s.lower() for s in tokenized]], is_tokenized=True)[0][
                        :len(tokenized) + 2]
        inf_label = {"*": [], "!": []}

        distance_matrix = 1 - sp.distance.cdist(inf_embedding, self.train_embedding, 'cosine')
        for i, word_distance in enumerate(distance_matrix):
            nearest = np.argmax(word_distance)
            if nearest in self.target_inx['*'] and word_distance[nearest] > self.threshold['*']:
                inf_label['*'].append(i)
            if nearest in self.target_inx['!'] and word_distance[nearest] > self.threshold['!']:
                inf_label['!'].append(i)
        print('\nInference.......................')
        self.print_sen(tokenized, inf_label)
        return tokenized, inf_label

    def print_sen(self, sentence_tokenized, label_dict):
        length = 0
        for i, w in enumerate(sentence_tokenized):
            if length + len(w) > 130:
                length = 0
                sys.stdout.write('\n')
            if length and w[0].isalnum() and sentence_tokenized[i - 1] not in "-'(\"":
                # add white space if w is not punctuation
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


if __name__ == "__main__":
    il = InstaLearn()
    while True:
        query = input('\ninput training data: ')
        il.train(query)
        query = input('\ninput inference data: ')
        _, _ = il.inference(query)
