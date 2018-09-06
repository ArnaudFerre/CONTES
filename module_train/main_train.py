#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: Training module to implement on ALVIS-ML/NLP
    If you want to cite this work in your publication or to have more details:
    http://www.aclweb.org/anthology/W17-2312.
Dependencies:
- PRONTO: https://pypi.python.org/pypi/pronto (MIT License: https://choosealicense.com/licenses/mit/)
    Maybe move shortly to Owlready (https://pypi.python.org/pypi/Owlready)
- Sklearn: Sklearn lib for ML (available with Anaconda) - Licence BSD
- Numpy: Numpy lib for scientific computing (available with Anaconda) - Licence BSD

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
from sklearn import linear_model
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

def getMatrix(dl_terms, vstTerm, dl_associations, vso, symbol="___"):
    """
    Description: Create the 2 training matrix in respect to their association. Each row of the matrix correspond to a
        vector (associated with a term or a concept). For the same number of a row, there is the vector of a term and
        for the other a vector of the associated concept ; (n_samples, n_features) and (n_sample, n_targets).
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :param vstTerm: VST with only the terms and the vectors contained in the dl_terms.
    :param dl_associations: The training set associating the id of terms and the if of concepts (NB: the respect of these
        IDs is the responsability of the user).
    :param vso: A VSO, that is a dictionary with id of concept (<XXX_xxxxxxxx: Label>) as keys and a numpy vector in value.
    :param symbol: Symbol delimiting the different token in a multi-words term.
    :return: Two matrix, one for the vectors of terms and another for the associated vectors of concepts.
    """

    nbTerms = len(dl_terms.keys())
    sizeVST = word2term.getSizeOfVST(vstTerm)
    sizeVSO = word2term.getSizeOfVST(vso)
    X_train = numpy.zeros((nbTerms, sizeVST))
    Y_train = numpy.zeros((nbTerms, sizeVSO))

    #stderr.write('sizeVST = %d\n' % sizeVST)
    #stderr.write('nbTerms = %d\n' % nbTerms)

    i = 0
    for id_term in dl_associations.keys():
        #stderr.write('id_term = %s\n' % str(id_term))
        #stderr.write('len(dl_associations[id_term]) = %d\n' % len(dl_associations[id_term]))
        for id_concept in dl_associations[id_term]:
            #stderr.write('id_concept = %s\n' % str(id_concept))
            termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
            #stderr.write('termForm = %s\n' % str(termForm))
            X_train[i] = vstTerm[termForm]
            Y_train[i] = vso[id_concept]
            i += 1
            break
    return X_train, Y_train



def train(vst_onlyTokens, dl_terms, dl_associations, ontology):
    """
    Description: Main module which calculates the regression parameters (a matrix)
    :param vst_onlyTokens: An initial VST containing only tokens and associated vectors.
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :param dl_associations: The training set associating the id of terms and the if of concepts (NB: the respect of these
        IDs is the responsability of the user).
    :param onto: A Pronto object representing an ontology.
    :return: the reg variable which contains two variables (coefficients matrix and error vectors). For more details, see:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        The reg variable enables to do prediction for the rest.
        Generate also a VSO represented by concepts and their vectors.
        To not lose this information, the unknown tokens contained in multiwords terms are also returned.
    """
    reg = linear_model.LinearRegression()
    # See parameters of the linear regression (fit_intercept, normalize, n_jobs)

    vso = onto.ontoToVec(ontology)

    vstTerm, l_unknownToken = word2term.wordVST2TermVST(vst_onlyTokens, dl_terms)

    X_train, Y_train = getMatrix(dl_terms, vstTerm, dl_associations, vso)

    reg.fit(X_train, Y_train)

    return reg, vso, l_unknownToken


def loadJSON(filename):
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        f = open(filename)
    result = json.load(f)
    f.close()
    return result;

class Train(OptionParser):
    def __init__(self):
        OptionParser.__init__(self, usage='usage: %prog [options]')
        self.add_option('--word-vectors', action='store', type='string', dest='word_vectors', help='path to word vectors file as produced by word2vec')
        self.add_option('--terms', action='append', type='string', dest='terms', help='path to terms file in JSON format (map: id -> array of tokens)')
        self.add_option('--attributions', action='append', type='string', dest='attributions', help='path to attributions file in JSON format (map: id -> array of concept ids)')
        self.add_option('--regression-matrix', action='append', type='string', dest='regression_matrix', help='path to the regression matrix file')
        self.add_option('--ontology', action='store', type='string', dest='ontology', help='path to ontology file in OBO format')
        self.add_option('--ontology-vector', action='store', type='string', dest='ontology_vector', help='path to the ontology vector file')
        
    def run(self):
        options, args = self.parse_args()
        if len(args) > 0:
            raise Exception('stray arguments: ' + ' '.join(args))
        if options.word_vectors is None:
            raise Exception('missing --word-vectors')
        if options.ontology is None:
            raise Exception('missing --ontology')
        if not options.terms:
            raise Exception('missing --terms')
        if not options.attributions:
            raise Exception('missing --attributions')
        if not options.regression_matrix:
            raise Exception('missing --regression-matrix')
        if len(options.terms) != len(options.attributions):
            raise Exception('there must be the same number of --terms and --attributions')
        if len(options.terms) != len(options.regression_matrix):
            raise Exception('there must be the same number of --terms and --regression-matrix')
        stderr.write('loading word embeddings: %s\n' % options.word_vectors)
        stderr.flush()
        word_vectors = loadJSON(options.word_vectors)
        stderr.write('loading ontology: %s\n' % options.ontology)
        stderr.flush()
        ontology = onto.loadOnto(options.ontology)
        first = True
        for terms_i, attributions_i, regression_matrix_i in zip(options.terms, options.attributions, options.regression_matrix):
            stderr.write('loading terms: %s\n' % terms_i)
            stderr.flush()
            terms = loadJSON(terms_i)
            stderr.write('loading attributions: %s\n' % attributions_i)
            stderr.flush()
            attributions = loadJSON(attributions_i)
            regression_matrix, ontology_vector, _ = train(word_vectors, terms, attributions, ontology)
            if options.ontology_vector is not None:
                # translate numpy arrays into lists
                serializable = dict((k, list(v)) for k, v in ontology_vector.iteritems())
                stderr.write('writing ontology vector: %s\n' % options.ontology_vector)
                stderr.flush()
                f = open(options.ontology_vector, 'w')
                json.dump(serializable, f)
                f.close()
            if first and options.regression_matrix is not None:
                first = False
                stderr.write('writing regression_matrix: %s\n' % regression_matrix_i)
                stderr.flush()
                joblib.dump(regression_matrix, regression_matrix_i)


            
if __name__ == '__main__':
    Train().run()
