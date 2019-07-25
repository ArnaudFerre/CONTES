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
import gensim
import sys, os
from optparse import OptionParser
import json
import gzip


sys.path.insert(0, os.path.abspath(".."))
from utils import word2term, onto

#######################################################################################################
# Functions
#######################################################################################################

def getMatrix(vstTerm, dl_terms, dl_associations, vso, symbol="___"):
    """
    Description: Create the 2 training matrix in respect to their association. Each row of the matrix correspond to a
        vector (associated with a term or a concept). For the same number of a row, there is the vector of a term and
        for the other a vector of the associated concept ; (n_samples, n_features) and (n_sample, n_targets).
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :param vstTerm: VST with only the terms and the vectors contained in the dl_terms.
    :param dl_associations: The training set associating the id of terms and the id of concepts (NB: the respect of these
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



def train(vstTerm, dl_terms, dl_associations, vso):
    """
    Description: Main module which calculates the regression parameters (a matrix)
    :param vstTerm: VST containing only vectors of terms.
    :param dl_associations: The training set associating the id of terms and the if of concepts (NB: the respect of these
        IDs is the responsability of the user).
    :param vso: A VSO, that is a dictionary with id of concept (<XXX_xxxxxxxx: Label>) as keys and a numpy vector in value..
    :return: the reg variable which contains two variables (coefficients matrix and error vectors). For more details, see:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        The reg variable enables to do prediction for the rest.
        To not lose this information, the unknown tokens contained in multiwords terms are also returned.
    """
    reg = linear_model.LinearRegression()
    # See parameters of the linear regression (fit_intercept, normalize, n_jobs)

    X_train, Y_train = getMatrix(vstTerm, dl_terms, dl_associations, vso)

    reg.fit(X_train, Y_train)

    return reg


def loadJSON(filename):
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        #f = open(filename, encoding='utf-8')
        f = open(filename)
    result = json.load(f)
    f.close()
    return result;

class Train(OptionParser):
    def __init__(self):
        OptionParser.__init__(self, usage='usage: %prog [options]')
        self.add_option('--terms', action='append', type='string', dest='terms', help='path to terms file in JSON format (map: id -> array of tokens)')
        self.add_option('--attributions', action='append', type='string', dest='attributions', help='path to attributions file in JSON format (map: id -> array of concept ids)')
        self.add_option('--ontology-vector', action='store', type='string', dest='vsoPath', help='path to the ontology vector file')
        self.add_option('--vst', action='append', type='string', dest='vstPath', help='path to terms vectors file in JSON format (map: token1___token2 -> array of floats)')
        self.add_option('--regression-matrix', action='append', type='string', dest='regression_matrix', help='path to the regression matrix file')

        
    def run(self):
        options, args = self.parse_args()
        if len(args) > 0:
            raise Exception('stray arguments: ' + ' '.join(args))
        if options.vsoPath is None:
            raise Exception('missing --ontology-vector')
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
        if options.vstPath is None:
            raise Exception('missing --vst')

        sys.stderr.write('loading expressions embeddings: %s\n' % options.vstPath)
        sys.stderr.flush()
        vstTerm = loadJSON(options.vstPath)

        sys.stderr.write('loading ontology-vector: %s\n' % options.vsoPath)
        sys.stderr.flush()
        vso = json.load(options.vsoPath)

        for terms_i, attributions_i, regression_matrix_i in zip(options.terms, options.attributions, options.regression_matrix):
            sys.stderr.write('loading terms: %s\n' % terms_i)
            sys.stderr.flush()
            terms = loadJSON(terms_i)
            sys.stderr.write('loading attributions: %s\n' % attributions_i)
            sys.stderr.flush()
            attributions = loadJSON(attributions_i)
            regression_matrix, _ = train(vstTerm, terms, attributions, vso)
            if options.regression_matrix is not None:
                sys.stderr.write('writing regression_matrix: %s\n' % regression_matrix_i)
                sys.stderr.flush()
                d = os.path.dirname(regression_matrix_i)
                if not os.path.exists(d) and d != '':
                    os.makedirs(d)
                joblib.dump(regression_matrix, regression_matrix_i)


if __name__ == '__main__':

    # Path to test data:
    mentionsFilePath = "../test/DATA/trainingData/terms_trainObo.json"
    attributionsFilePath = "../test/DATA/trainingData/attributions_trainObo.json"
    vst_termsPath = "../test/DATA/expressionEmbeddings/vstTerm_trainObo.json"
    SSO_path = "../test/DATA/VSO_OntoBiotope_BioNLP-ST-2016.json"
    regmatPath = "../test/DATA/learnedHyperparameters/model.sav"

    # load training data:
    print("\nLoading training data...")
    extractedMentionsFile = open(mentionsFilePath, 'r')
    dl_trainingTerms = json.load(extractedMentionsFile)
    attributionsFile = open(attributionsFilePath, 'r')
    attributions = json.load(attributionsFile)
    vstTermsFile = open(vst_termsPath, 'r')
    vst_terms = json.load(vstTermsFile)
    print("Training data loaded.\n")

    # Building of concept embeddings and training:
    print("Loading Semantic Space of the Ontology (SSO)...")
    SSO_file = open(SSO_path, "r")
    SSO = json.load(SSO_file)
    print("SSO loaded.\n")

    print("Training of the CONTES method...")
    regMat = train(vst_terms, dl_trainingTerms, attributions, SSO)
    print("Training done.\n")

    print("Saving learned hyperparameters...")
    joblib.dump(regMat, regmatPath)
    print("Saving done.")


    #Train().run()
