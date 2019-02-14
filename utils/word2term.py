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


#######################################################################################################
# Functions
#######################################################################################################
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

#######################################################################################################
# Tests
#######################################################################################################
if __name__ == '__main__':

    print("Test of wordVecToTermVec...")

    sizeVst = 2
    vst_onlyTokens = {
        "fear":numpy.random.rand(sizeVst), "is":numpy.random.rand(sizeVst), "the":numpy.random.rand(sizeVst),
        "path":numpy.random.rand(sizeVst), "to":numpy.random.rand(sizeVst), "dark":numpy.random.rand(sizeVst),
        "side":numpy.random.rand(sizeVst), "leads":numpy.random.rand(sizeVst), "anger":numpy.random.rand(sizeVst),
        "hate":numpy.random.rand(sizeVst), "suffering":numpy.random.rand(sizeVst)
    }

    dl_terms = {"01" : ["dark", "side"], "02" : ["fear"], "03" : ["light", "side"], "04" : ["force"] }

    vstTerm, l_unknownToken = wordVST2TermVST(vst_onlyTokens, dl_terms)
    print("VST with terms vectors: " + str(vstTerm))
    print("Unknown Tokens: " + str(l_unknownToken))

    print("Test of wordVecToTermVec end.")
