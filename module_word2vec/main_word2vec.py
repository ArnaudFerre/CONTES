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

import gensim
import json
import numpy
from sys import stderr, stdin
from optparse import OptionParser

class Word2Vec(OptionParser):
    def __init__(self):
        OptionParser.__init__(self, usage='usage: %prog [options]')
        self.add_option('--json', action='store', type='string', dest='json', help='JSON output filename')
        self.add_option('--txt', action='store', type='string', dest='txt', help='TXT output filename')
        self.add_option('--min-count', action='store', type='int', dest='minCount', default=0, help='Ignore all words with total frequency lower than this')
        self.add_option('--vector-size', action='store', type='int', dest='vectSize', default=300, help='The dimensionality of the feature vectors, often effective between 100 and 300')
        self.add_option('--workers', action='store', type='int', dest='workerNum', default=2, help='Use this many worker threads to train the model (=faster training with multicore machines)')
        self.add_option('--skip-gram', action='store_true', dest='skipGram', default=False, help='Defines the training algorithm, by default CBOW is used, otherwise skip-gram is employed')
        self.add_option('--window-size', action='store', type='int', dest='windowSize', default=2, help='The maximum distance between the current and predicted word within a sentence')
        self.corpus = []

    def buildVector(self, workerNum=8, minCount=0, vectSize=200, skipGram=True, windowSize=2, learningRate=0.05, numIteration=5, negativeSampling=5, subSampling=0.001):
        """
        Description: Implementation of the neuronal method Word2Vec to create word vectors based on the distributional
        semantics hypothesis.
        
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
        model = gensim.models.Word2Vec(self.corpus, min_count=minCount, size=vectSize, workers=workerNum, sg=skipGram,
                                       window=windowSize, alpha=learningRate, iter=numIteration, negative=negativeSampling,
                                       sample=subSampling)
        self.VST = dict((k, list(numpy.float_(npf32) for npf32 in model.wv[k])) for k in model.wv.vocab.keys())

    def run(self):
        options, args = self.parse_args()
        self.readCorpusFiles(args)
        self.buildVector(minCount=options.minCount, vectSize=options.vectSize, workerNum=options.workerNum, skipGram=options.skipGram, windowSize=options.windowSize)
        self.writeJSON(options.json)
        self.writeTxt(options.txt)
        
    def writeJSON(self, fileName):
        if fileName is None:
            return
        f = open(fileName, 'w')
        f.write(json.dumps(self.VST))
        f.close()

    def writeTxt(self, fileName):
        f = open(fileName, 'w')
        for k, v in self.VST.iteritems():
            f.write(k)
            f.write('\t')
            f.write(str(v))
            f.write('\n')
        f.close()
        
    def readCorpusFiles(self, fileNames):
        if len(fileNames) == 0:
            self.readCorpus(stdin)
            return
        for fn in fileNames:
            f = open(fn)
            self.readCorpus(f)
            f.close()
            
    def readCorpus(self, f):
        current_sentence = []
        for line in f:
            line = line.strip()
            if line == '':
                if len(current_sentence) > 0:
                    self.corpus.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
        if len(current_sentence) > 0:
            self.corpus.append(current_sentence)

if __name__ == '__main__':
    Word2Vec().run()
