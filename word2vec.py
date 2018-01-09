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

    :return: evt is a dictionary containing the form of token as key and the corresponding vector as unique value.

    For more details, see: https://radimrehurek.com/gensim/models/word2vec.html
    """

    evt = dict()

    # train word2vec on the sentences
    model = gensim.models.Word2Vec(ll_corpus, min_count=minCount, size=vectSize, workers=workerNum, sg=skipGram,
                                   window=windowSize, alpha=learningRate, iter=numIteration, negative=negativeSampling,
                                   sample=subSampling)

    for wordForm in model.wv.vocab.keys():
        evt[wordForm] = model.wv[wordForm]

    return evt


#######################################################################################################
# Tests
#######################################################################################################
if __name__ == '__main__':

    print("Test of Word2Vec/Gensim...")

    ll_testCorpus=[
        ["fear", "is", "the", "path", "to", "the", "dark", "side", "."],
        ["fear", "leads", "to", "anger", "."],
        ["anger", "leads", "to", "hate", "."],
        ["hate", "leads", "to", "suffering", "."]
    ]

    testEVT = WordsVectorization(ll_testCorpus, minCount=0, vectSize=2, workerNum=8, skipGram=True, windowSize=2)

    print("Vocabulary: "+str(testEVT.keys()))

    print("Test of Word2Vec/Gensim end.")
