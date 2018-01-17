#!/usr/bin/env python
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
import pronto
from sklearn import linear_model
import numpy

from utils import word2term

#######################################################################################################
# Functions
#######################################################################################################
def loadOnto(ontoPath):
    """
    Description: Load an ontology object from a specified path.
    :param ontoPath: path of the ontology
    :return: pronto object representing an ontology
    NB: Even if it should accept OWL file, only OBO seems work.
    """
    onto = pronto.Ontology(ontoPath)
    return onto


def ontoToVec(onto):
    """
    Description: Create a vector space of the ontology. It uses hierarchical information to do this.
    :param onto: A Pronto object representing an ontology.
    :return: A VSO, that is a dictionary with id of concept (<XXX_xxxxxxxx: Label>) as keys and a numpy vector in value.
    """
    vso = dict()

    size = len(onto)
    d_assoDim = dict()

    for i, concept in enumerate(onto):
        id_concept = str(concept)
        vso[id_concept] = numpy.zeros(size)
        d_assoDim[id_concept] = i
        vso[id_concept][d_assoDim[id_concept]] = 1

    for concept in onto:
        id_concept = str(concept)
        for parent in concept.rparents(-1, True):
            id_parent = str(parent)
            vso[id_concept][d_assoDim[id_parent]] = 1

    return vso


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

    i = 0
    for id_term in dl_associations.keys():
        for id_concept in dl_associations[id_term]:
            termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
            X_train[i] = vstTerm[termForm]
            Y_train[i] = vso[id_concept]
            i += 1

    return X_train, Y_train



def train(vst_onlyTokens, dl_terms, dl_associations, onto):
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

    vso = ontoToVec(onto)

    vstTerm, l_unknownToken = word2term.wordVST2TermVST(vst_onlyTokens, dl_terms)

    X_train, Y_train = getMatrix(dl_terms, vstTerm, dl_associations, vso)

    reg.fit(X_train, Y_train)

    return reg, vso, l_unknownToken





# TO TRASH (just for test)

def getCosSimilarity(vec1, vec2):
    from scipy import spatial
    result = 1 - spatial.distance.cosine(vec1, vec2)
    return result

def getNearearConcept(vecTerm, vso):
    max = 0
    mostSimilarConcept = None
    for id_concept in vso.keys():
        dist = getCosSimilarity(vecTerm, vso[id_concept])
        if dist > max:
            max = dist
            mostSimilarConcept = id_concept
    return mostSimilarConcept

def testPredict(reg, vst_onlyTokens, dl_terms, symbol="___"):

    vstTerm, l_unknownToken = word2term.wordVST2TermVST(vst_onlyTokens, dl_terms)

    result = dict()

    vsoTerms = dict()
    for id_term in dl_terms.keys():
        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        x = vstTerm[termForm].reshape(1, -1)
        vsoTerms[termForm] = reg.predict(x)[0]

        result[termForm] = getNearearConcept(vsoTerms[termForm], vso)

    return vsoTerms, result


#######################################################################################################
# Tests
#######################################################################################################
if __name__ == '__main__':

    print("Test of train module...")

    # Test data :

    sizeVst = 2
    vst_onlyTokens = {
        "dog" : numpy.random.rand(sizeVst), "neighbours": numpy.random.rand(sizeVst), "cat": numpy.random.rand(sizeVst),
        "lady": numpy.random.rand(sizeVst), "tramp": numpy.random.rand(sizeVst), "fear": numpy.random.rand(sizeVst),
        "path": numpy.random.rand(sizeVst), "dark": numpy.random.rand(sizeVst), "side": numpy.random.rand(sizeVst),
        "leads": numpy.random.rand(sizeVst), "anger": numpy.random.rand(sizeVst), "hate": numpy.random.rand(sizeVst),
        "yoda": numpy.random.rand(sizeVst)
    }

    l_term1 = ["dog", "of", "my", "neighbours"]
    l_term2 = ["cat", "from", "lady", "and", "the", "tramp"]
    dl_terms = {"01": l_term1, "02": l_term2, "03" : ["dog"], "04": ["yoda"]}

    dl_associations = {
        "01" : ["<SDO_00000001: Dog>"],
        "02" : ["<SDO_00000003: Siamoise>"],
        "03" : ["<SDO_00000001: Dog>"],
        "04": ["<SDO_00000000: Animalia>"]
    }

    ontoPath = "../../testOnto.obo"
    ontoTest = loadOnto(ontoPath)


    # Module test :

    vso = ontoToVec(ontoTest)
    print("VSO : " + str(vso))

    reg, vso, l_unknownToken = train(vst_onlyTokens, dl_terms, dl_associations, ontoTest)
    print("VSO : " + str(vso))
    print("Unknown Tokens: " + str(l_unknownToken))

    # NB : Use joblib (from sklearn.externals import joblib) to save a file for reg.



    # Prediction test on training set: (TO TRASH)
    print "\n"
    vsoTerms, result = testPredict(reg, vst_onlyTokens, dl_terms, symbol="___")
    print("Terms in VSO : " + str(vsoTerms))
    import json
    print(json.dumps(result, indent=4))


    print("Test of train module end.")