#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: Word2Vec/Gensim module to implement on ALVIS-ML/NLP
Dependencies: Require the installation of Gensim (https://radimrehurek.com/gensim/install.html)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


#######################################################################################################
# Import modules & set up logging
#######################################################################################################
import gensim
import json
import numpy
from sys import stderr, stdin
from optparse import OptionParser


#######################################################################################################
# Function
#######################################################################################################
def WordsVectorization(ll_corpus, workerNum=8,
                       minCount=0, vectSize=200,  skipGram=True, windowSize=2,
                       learningRate=0.05, numIteration=5, negativeSampling=5, subSampling=0.001):
    """
    Description: Implementation of the neuronal method Word2Vec to create word vectors based on the distributional
        semantics hypothesis.

    :param ll_corpus: Corpus as a list of list of tokens; Each list of tokens represents a sentence.
        Other preprocesses may be required to improve the results (better segmentation, lower case, lemmatization, ...)
    :param workerNum: Use this many worker threads to train the model (=faster training with multicore machines).

    :param minCount: Ignore all words with total frequency lower than this.
    :param vectSize: The dimensionality of the feature vectors. Often effective between 100 and 300.
    :param skipGram: Defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
    :param windowSize: The maximum distance between the current and predicted word within a sentence.

    :param learningRate: Alpha is the initial learning rate (will linearly drop to min_alpha as training progresses).
    :param numIteration: Number of iterations (epochs) over the corpus. Default is 5.
    :param negativeSampling: if > 0, negative sampling will be used, the int for negative specifies how many
        “noise words” should be drawn (usually between 5-20). Default is 5. If set to 0, no negative samping is used.
    :param subSampling: Threshold for configuring which higher-frequency words are randomly downsampled;
        default is 1e-3, useful range is (0, 1e-5).

    :return: vst (vector space of terms) is a dictionary containing the form of token as key and the corresponding
        vector as unique value.

    For more details, see: https://radimrehurek.com/gensim/models/word2vec.html
    """
    model = gensim.models.Word2Vec(ll_corpus, min_count=minCount, size=vectSize, workers=workerNum, sg=skipGram,
                                   window=windowSize, alpha=learningRate, iter=numIteration, negative=negativeSampling,
                                   sample=subSampling)
    return dict((k, _to_float_array(model.wv[k])) for k in model.wv.vocab.keys())


def _to_float_array(npa):
    return list(numpy.float_(npf32) for npf32 in npa)

def read_corpus(f, corpus):
    current_sentence = []
    for line in f:
        line = line.strip()
        if line == '':
            if len(current_sentence) > 0:
                corpus.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(line)
    if len(current_sentence) > 0:
        corpus.append(current_sentence)

class Word2Vec(OptionParser):
    def __init__(self):
        OptionParser.__init__(self, usage='usage: %prog [options]')
        self.add_option('--json', action='store', type='string', dest='json', help='JSON output filename')
        self.add_option('--txt', action='store', type='string', dest='txt', help='TXT output filename')

    def run(self):
        options, args = self.parse_args()
        corpus = []
        for fn in args:
            f = open(fn)
            read_corpus(f, corpus)
            f.close()
        if len(args) == 0:
            read_corpus(stdin, corpus)
        VST = WordsVectorization(corpus, minCount=0, vectSize=2, workerNum=8, skipGram=True, windowSize=2)
        if options.json is not None:
            f = open(options.json, 'w')
            f.write(json.dumps(VST))
            f.close()
        if options.txt is not None:
            f = open(options.txt, 'w')
            for k, v in VST.iteritems():
                f.write(k)
                f.write('\t')
                f.write(str(list(numpy.float_(v))))
                f.write('\n')
            f.close()

if __name__ == '__main__':
    Word2Vec().run()
