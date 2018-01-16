#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description:
Dependencies:

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


#######################################################################################################
# Functions
#######################################################################################################
def getSizeOfEVT(evt):
    """
    Description: Return the size of the vector space of an evt variable.
    Look for the first existing vector of the evt and get its size.
    NB: Used only to not have to pass the size of the evt as a parameter.
    """
    size = 0
    for key in evt:
        if evt[key] is not None:
            size = evt[key].size
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


def calculateTermVec(evt_onlyTokens, l_tokens, l_unknownToken):
    """
    Description: Calculates a vector for a multi-words (or single-word) term from the vectors of tokens forming the term.
    :param evt_onlyTokens: An EVT, but normally containing only vector for tokens.
    :param l_tokens: List of orderly tokens forming a term.
    :param l_unknownToken: If a token of the term doesn't have calculated vector in the evt, a register is saved here.
    :return: A vector for the term given by l_tokens. If there is an unique token and that this one is not in the evt,
    then a null vector is associated to this one-word term.
    """
    size = getSizeOfEVT(evt_onlyTokens)
    vec = numpy.zeros((size))

    numberOfKnownTokens = 0

    for i, token in enumerate(l_tokens):
        if token in evt_onlyTokens.keys():
            vec += evt_onlyTokens[token]
            numberOfKnownTokens += 1
        else:
            l_unknownToken.append(token)

    if numberOfKnownTokens > 0:
        vec = vec / numberOfKnownTokens

    return vec, l_unknownToken


def wordEVT2TermEVT(evt_onlyTokens, dl_terms):
    """
    Description: Generate a new EVT in which all tokens and all terms from the dl_terms is exprimed.
    :param evt_onlyTokens: An EVT, but normally containing only vector for tokens.
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :return: A new EVT.
    """
    evt = dict()
    l_unknownToken = list()

    for id_term in dl_terms.keys():

        term = getFormOfTerm(dl_terms[id_term])
        vec, l_unknownToken = calculateTermVec(evt_onlyTokens, dl_terms[id_term], l_unknownToken)

        evt[term] = vec

    return evt, l_unknownToken

#######################################################################################################
# Tests
#######################################################################################################
if __name__ == '__main__':

    print("Test of wordVecToTermVec...")

    sizeEvt = 2
    evt_onlyTokens = {
        "fear":numpy.random.rand(sizeEvt), "is":numpy.random.rand(sizeEvt), "the":numpy.random.rand(sizeEvt),
        "path":numpy.random.rand(sizeEvt), "to":numpy.random.rand(sizeEvt), "dark":numpy.random.rand(sizeEvt),
        "side":numpy.random.rand(sizeEvt), "leads":numpy.random.rand(sizeEvt), "anger":numpy.random.rand(sizeEvt),
        "hate":numpy.random.rand(sizeEvt), "suffering":numpy.random.rand(sizeEvt)
    }

    dl_terms = {"01" : ["dark", "side"], "02" : ["fear"], "03" : ["light", "side"], "04" : ["force"] }

    evtTerm, l_unknownToken = wordEVT2TermEVT(evt_onlyTokens, dl_terms)
    print(evtTerm)
    print("Unknown Tokens: " + str(l_unknownToken))

    print("Test of wordVecToTermVec end.")
