#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: If you have trained the module_train on a training set (terms associated with concept(s)), you can do here
    a prediction of normalization with a test set (new terms without pre-association with concept). NB : For now, you
    can only use a Sklearn object from the class LinearRegression.
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
from sklearn.externals import joblib
import numpy
from sys import stderr, stdin
from optparse import OptionParser
from utils import word2term, onto
import json
import gzip

#######################################################################################################
# Functions
#######################################################################################################
def getCosSimilarity(vec1, vec2):
    """
    Description: Calculates the cosine similarity between 2 vectors.
    """
    from scipy import spatial
    result = 1 - spatial.distance.cosine(vec1, vec2)
    return result


def getNearestConcept(vecTerm, vso):
    """
    Description: For now, calculates all the cosine similarity between a vector and the concept-vectors of the VSO,
        then, gives the nearest.
    :param vecTerm: A vector in the VSO.
    :param vso: A VSO (dict() -> {"id" : [vector], ...}
    :return: the id of the nearest concept.
    """
    maxsim = 0
    mostSimilarConcept = None
    for id_concept in vso.keys():
        sim = getCosSimilarity(vecTerm, vso[id_concept])
        if sim > maxsim:
            maxsim = sim
            mostSimilarConcept = id_concept
    return mostSimilarConcept, maxsim

def predictor(vst_onlyTokens, dl_terms, vso, transformationParam, symbol="___"):
    """
    Description: From a calculated linear projection from the training module, applied it to predict a concept for each
        terms in parameters (dl_terms).
    :param vst_onlyTokens: An initial VST containing only tokens and associated vectors.
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :param vso: A VSO (dict() -> {"id" : [vector], ...}
    :param transformationParam: LinearRegression object from Sklearn. Use the one calculated by the training module.
    :param symbol: Symbol delimiting the different token in a multi-words term.
    :return: A list of tuples containing : ("term form", "term id", "predicted concept id") and a list of unknown tokens
        containing in the terms from dl_terms.
    """
    lt_predictions = list()

    vstTerm, l_unknownToken = word2term.wordVST2TermVST(vst_onlyTokens, dl_terms)

    result = dict()

    vsoTerms = dict()
    for id_term in dl_terms.keys():
        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        x = vstTerm[termForm].reshape(1, -1)
        vsoTerms[termForm] = transformationParam.predict(x)[0]

        result[termForm] = getNearestConcept(vsoTerms[termForm], vso)

    for id_term in dl_terms.keys():
        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        cat, sim = result[termForm]
        prediction = (termForm, id_term, cat, sim)
        lt_predictions.append(prediction)

    return lt_predictions, l_unknownToken


def loadJSON(filename):
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        f = open(filename, encoding='utf-8')
    result = json.load(f)
    f.close()
    return result;


class Predictor(OptionParser):
    def __init__(self):
        OptionParser.__init__(self, usage='usage: %prog [options]')
        self.add_option('--word-vectors', action='store', type='string', dest='word_vectors', help='path to word vectors file as produced by word2vec')
        self.add_option('--ontology', action='store', type='string', dest='ontology', help='path to ontology file in OBO format')
        self.add_option('--terms', action='append', type='string', dest='terms', help='path to terms file in JSON format (map: id -> array of tokens)')
        self.add_option('--regression-matrix', action='append', type='string', dest='regression_matrix', help='path to the regression matrix file as produced by the training module')
        self.add_option('--output', action='append', type='string', dest='output', help='file where to write predictions')
        
    def run(self):
        options, args = self.parse_args()
        if len(args) > 0:
            raise Exception('stray arguments: ' + ' '.join(args))
        if options.word_vectors is None:
            raise Exception('missing --word-vectors')
        if options.ontology is None:
            raise Exception('missing --ontology')
        if not(options.terms):
            raise Exception('missing --terms')
        if not(options.regression_matrix):
            raise Exception('missing --regression-matrix')
        if not(options.output):
            raise Exception('missing --output')
        if len(options.terms) != len(options.regression_matrix):
            raise Exception('there must be the same number of --terms and --regression-matrix')
        if len(options.terms) != len(options.output):
            raise Exception('there must be the same number of --terms and --output')
        stderr.write('loading word embeddings: %s\n' % options.word_vectors)
        stderr.flush()
        word_vectors = loadJSON(options.word_vectors)
        stderr.write('loading ontology: %s\n' % options.ontology)
        stderr.flush()
        ontology = onto.loadOnto(options.ontology)
        vso = onto.ontoToVec(ontology)
        for terms_i, regression_matrix_i, output_i in zip(options.terms, options.regression_matrix, options.output):
            stderr.write('loading terms: %s\n' % terms_i)
            stderr.flush()
            terms = loadJSON(terms_i)
            stderr.write('loading regression matrix: %s\n' % regression_matrix_i)
            stderr.flush()
            regression_matrix = joblib.load(regression_matrix_i)
            stderr.write('predicting\n')
            stderr.flush()
            prediction, _ = predictor(word_vectors, terms, vso, regression_matrix)
            stderr.write('writing predictions: %s\n' % output_i)
            stderr.flush()
            f = open(output_i, 'w')
            for _, term_id, concept_id, similarity in prediction:
                f.write('%s\t%s\t%f\n' % (term_id, concept_id, similarity))
            f.close()

if __name__ == '__main__':
    Predictor().run()
