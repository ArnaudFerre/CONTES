#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: Giving a list a new terms (notably multiwords terms) and a VST from a corpus, embeds the new terms in the
    current VST (currently with calculating the barycentre of tokens composing a multiwords term).
    If you want to cite this work in your publication or to have more details:
    http://www.aclweb.org/anthology/W17-2312.
Dependency: Numpy lib (available with Anaconda)

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
import numpy
from optparse import OptionParser
from sys import stderr, stdin
import json
import gzip
import gensim
import os


#######################################################################################################
# Functions
#######################################################################################################
def loadJSON(filename):
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        #f = open(filename, encoding='utf-8')
        f = open(filename)
    result = json.load(f)
    f.close()
    return result;


def getSizeOfVST(vst):
    """
    Description: Return the size of the vector space of an vst variable.
    Look for the first existing vector of the vst and get its size.
    NB: Used only to not have to pass the size of the vst as a parameter.
    """
    size = 0
    for key in vst:
        if vst[key] is not None:
            #size = vst[key].size
            size = len(vst[key])
            break
    return size


def getFormOfTerm(l_tokens, symbol="___"):
    """
    Description: Enable to save a specific form of a multiwords term, which could be segmented later on.
    :param l_tokens: List of orderly tokens forming a term.
    :param symbol: Symbol delimiting the different token in a multi-words term.
    :return: A string representing a term with a symbol (default is "___") between their tokens.
    """
    term = ""
    for i, token in enumerate(l_tokens):
        if i == len(l_tokens) - 1:
            term += token
        else:
            term += token + symbol
    return term


def calculateTermVec(vst_onlyTokens, l_tokens, l_unknownToken):
    """
    Description: Calculates a vector for a multi-words (or single-word) term from the vectors of tokens forming the term.
    :param vst_onlyTokens: An VST, but normally containing only vector for tokens.
    :param l_tokens: List of orderly tokens forming a term.
    :param l_unknownToken: If a token of the term doesn't have calculated vector in the vst, a register is saved here.
    :return: A vector for the term given by l_tokens. If there is an unique token and that this one is not in the vst,
    then a null vector is associated to this one-word term.
    """
    size = getSizeOfVST(vst_onlyTokens)
    vec = numpy.zeros((size))

    numberOfKnownTokens = 0

    for i, token in enumerate(l_tokens):

        if token in vst_onlyTokens.keys():
            vec += vst_onlyTokens[token]
            numberOfKnownTokens += 1
        else:
            if token not in l_unknownToken:
                l_unknownToken.append(token)

    if numberOfKnownTokens > 0:
        vec = vec / numberOfKnownTokens

    return vec, l_unknownToken


def wordVST2TermVST(vst_onlyTokens, dl_terms):
    """
    Description: Generate a new VST variable in which all terms from the dl_terms is expressed.
        Mathematically speaking, it is the same vector space than vst_onlyTokens, but the returned variable doesn't
        conserve the information of the tokens.
    :param vst_onlyTokens: An VST, but normally containing only vector for tokens.
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :return: A new VST.
    """
    vst = dict()
    l_unknownToken = list()

    for id_term in dl_terms.keys():

        term = getFormOfTerm(dl_terms[id_term])
        vec, l_unknownToken = calculateTermVec(vst_onlyTokens, dl_terms[id_term], l_unknownToken)

        vst[term] = vec

    return vst, l_unknownToken



class WordVec2expVec(OptionParser):
    def __init__(self):
        OptionParser.__init__(self, usage='usage: %prog [options]')

        # Input:
        self.add_option('--word-vectors', action='store', type='string', dest='word_vectors', help='path to word vectors JSON file as produced by word2vec')
        self.add_option('--word-vectors-bin', action='store', type='string', dest='word_vectors_bin', help='path to word vectors binary file as produced by word2vec')
        self.add_option('--terms', action='append', type='string', dest='terms', help='path to terms file in JSON format (map: id -> array of tokens)')

        # Output:
        self.add_option('--vst', action='append', type='string', dest='vstPath', help='path to terms vectors file in JSON format (map: token1___token2 -> array of floats)')


    def run(self):
        options, args = self.parse_args()

        if len(args) > 0:
            raise Exception('stray arguments: ' + ' '.join(args))
        if options.word_vectors is None and options.word_vectors_bin is None:
            raise Exception('missing either --word-vectors or --word-vectors-bin')
        if options.word_vectors is not None and options.word_vectors_bin is not None:
            raise Exception('incompatible --word-vectors or --word-vectors-bin')
        if not options.terms:
            raise Exception('missing --terms')
        if not options.vst:
            raise Exception('missing --vst')

        if options.word_vectors is not None:
            stderr.write('loading word embeddings: %s\n' % options.word_vectors)
            stderr.flush()
            word_vectors = loadJSON(options.word_vectors)
        elif options.word_vectors_bin is not None:
            stderr.write('loading word embeddings: %s\n' % options.word_vectors_bin)
            stderr.flush()
            model = gensim.models.Word2Vec.load(options.word_vectors_bin)
            word_vectors = dict((k, list(numpy.float_(npf32) for npf32 in model.wv[k])) for k in model.wv.vocab.keys())

        stderr.write('loading terms\n')
        stderr.flush()
        dl_terms = loadJSON(options.terms)

        vstTerms, _ = wordVST2TermVST(word_vectors, dl_terms)

        if options.vstPath is not None:
            stderr.write('writing vst of expressions: %s\n' % options.vstPath)
            stderr.flush()
            VST = dict()
            for term in vstTerms.keys():
                VST[term] = list(numpy.float_(npf32) for npf32 in vstTerms[term])
            del vstTerms
            f = open(options.vstPath, 'w')
            json.dump(VST, f)
            f.close()



if __name__ == '__main__':
    #WordVec2expVec().run()

    # Path to test data:
    mentionsFilePath = "../test/DATA/trainingData/terms_trainObo.json"
    modelPath = "../test/DATA/wordEmbeddings/VST_count0_size100_iter50.model"  # the provided models are really small models, just to test execution
    vstTerms_path = "../test/DATA/expressionEmbeddings/vstTerm_trainObo.json"

    # Load an existing W2V model (Gensim format):
    print("\nLoading word embeddings...")
    from gensim.models import Word2Vec
    filename, file_extension = os.path.splitext(modelPath)
    model = Word2Vec.load(modelPath)
    vst_onlyTokens = dict((k, list(numpy.float_(npf32) for npf32 in model.wv[k])) for k in model.wv.vocab.keys())
    print("Word embeddings loaded.\n")

    print("\nLoading some expressions (for training dataset)...")
    extractedMentionsFile = open(mentionsFilePath, 'r')
    dl_trainingTerms = json.load(extractedMentionsFile)
    print("Expressions loaded.\n")

    print("Calculating representions for expressions (possibly multiwords)...")
    vstTerms, l_unknownToken = wordVST2TermVST(vst_onlyTokens, dl_trainingTerms)
    print(l_unknownToken)
    VST = dict()
    for term in vstTerms.keys():
        VST[term] = list(numpy.float_(npf32) for npf32 in vstTerms[term])
    del vstTerms
    print("Calculating representions for expressions done.\n")

    print("Writing of VST of expressions...")
    f = open(vstTerms_path, 'w')
    json.dump(VST, f)
    f.close()
    print("VST has been written.\n")