#!/usr/bin/env python
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
import numpy

from utils import word2term

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
    max = 0
    mostSimilarConcept = None
    for id_concept in vso.keys():
        dist = getCosSimilarity(vecTerm, vso[id_concept])
        if dist > max:
            max = dist
            mostSimilarConcept = id_concept
    return mostSimilarConcept



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
        prediction = (termForm, id_term, result[termForm])
        lt_predictions.append(prediction)

    return lt_predictions, l_unknownToken

#######################################################################################################
# Tests
#######################################################################################################
if __name__ == '__main__':
    print("Test of predictor module...")

    # Test data :

    sizeVst = 2
    vst_onlyTokens = {
        "dog": numpy.random.rand(sizeVst), "neighbours": numpy.random.rand(sizeVst), "cat": numpy.random.rand(sizeVst),
        "lady": numpy.random.rand(sizeVst), "tramp": numpy.random.rand(sizeVst), "fear": numpy.random.rand(sizeVst),
        "path": numpy.random.rand(sizeVst), "dark": numpy.random.rand(sizeVst), "side": numpy.random.rand(sizeVst),
        "leads": numpy.random.rand(sizeVst), "anger": numpy.random.rand(sizeVst), "hate": numpy.random.rand(sizeVst),
        "yoda": numpy.random.rand(sizeVst)
    }

    l_term1 = ["dog", "of", "my", "neighbours"]
    l_term2 = ["cat", "from", "lady", "and", "the", "tramp"]
    dl_termsTrain = {"01": l_term1, "02": l_term2, "03" : ["dog"], "04": ["yoda"], "05": ["cat"]}

    dl_associations = {
        "01": ["<SDO_00000001: Dog>"],
        "02": ["<SDO_00000003: Siamoise>"],
        "03": ["<SDO_00000001: Dog>"],
        "04": ["<SDO_00000000: Animalia>"],
        "05": ["<SDO_00000002: Cat>"]
    }


    from module_train import main_train
    ontoPath = "testOnto.obo"
    ontoTest = main_train.loadOnto(ontoPath)

    vso = main_train.ontoToVec(ontoTest)
    print("VSO : " + str(vso))

    reg, vso, l_unknownToken = main_train.train(vst_onlyTokens, dl_termsTrain, dl_associations, ontoTest)
    print("VSO : " + str(vso))
    print("Unknown Tokens: " + str(l_unknownToken) + "\n\n")



    # Module test :

    l_term1 = ["friendly", "little", "dog"]
    l_term2 = ["dark", "side", "of", "the", "moon"]
    l_term3 = ["dog", "from", "lady", "and", "the", "tramp"]
    dl_termsTest = {"01": l_term1, "02": l_term2, "03": ["dog"], "04": ["cat"], "05": l_term3}

    lt_predictions, l_unknownToken = predictor(vst_onlyTokens, dl_termsTest, vso, reg, symbol="___")
    print("Predictions: " + str(lt_predictions))
    print("Unknown Tokens for test: " + str(l_unknownToken) + "\n\n")

    print("Test of predictor module end.")