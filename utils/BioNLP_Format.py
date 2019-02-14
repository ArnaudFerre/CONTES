#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: Utilitary functions to automaticcaly parse and write A1 & A2 files.
        For more details on BioLP-ST formats, see: http://2011.bionlp-st.org/home/file-formats

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
import os, numpy, re
import nltk
from sklearn.externals import joblib
import gensim

from module_train import main_train
from word2term import wordVST2TermVST
from module_predictor import main_predictor

#######################################################################################################
# Functions
#######################################################################################################
def loadPredictionFile(filepath):
    dt_prediction = dict()
    dl_terms = dict()

    file = open(filepath, "r")

    id = 0
    for line in file:
        term = line.split("\t")[0]

        l_term = nltk.tokenize.word_tokenize(term)
        termID = id
        id += 1
        dl_terms[termID] = l_term

        conceptID = line.split("\t")[1]
        conceptLabel = line.split("\t")[2].split("\n")[0]

        dt_prediction[termID] = (term, conceptID, conceptLabel)

    file.close()

    return dt_prediction, dl_terms


def adaptConceptsIDsFromPrediction(dt_prediction):

    for term in dt_prediction.keys():
        c_conceptID = dt_prediction[term]
        adaptedConceptName = ["<" + c_conceptID[1] + ": " + c_conceptID[2] + ">"]
        dt_prediction[term] = adaptedConceptName

    return dt_prediction


#####################################################################################################################
# Final creation of a2 files
#####################################################################################################################
def loadA1File(A1File):
    dd_a1 = dict()

    fichier = open(A1File)
    for line in fichier:
        l_line = line.split()
        term = ""
        if l_line[1] == "Habitat" or l_line[1] == "Bacteria":

            # Car parfois plusieurs occurences d'un termes (ex : T14	Bacteria 879 906;922 927	R. conorii reference strain no. 7)
            nbOcc = len(line.split(";"))
            splitSwitch = nbOcc - 1

            for i in range(len(l_line)):
                if i >= 4 + splitSwitch:
                    if i == (len(l_line) - 1):
                        term += l_line[i]
                    else:
                        term += l_line[i] + " "

            if term not in dd_a1.keys():
                dd_a1[term] = dict()

            dd_a1[term]["cat"] = l_line[1]
            #dd_a1[term]["tokens"] = getTokenizedTermsWithoutStopWords(term.lower())

            if "T" not in dd_a1[term].keys():
                dd_a1[term]["T"] = list()
                dd_a1[term]["T"].append(l_line[0])
            else:
                dd_a1[term]["T"].append(l_line[0])

    return dd_a1


def parseA1(a1Path):
    ddd_a1 = dict()

    files = os.listdir(a1Path)

    for file in files:
        filename, file_extension = os.path.splitext(file)

        if file_extension == ".a1":
            ddd_a1[filename] = loadA1File(a1Path + "/" + file)

    return ddd_a1


def getTermsFromA1(ddd_a1):
    dl_terms = dict()
    errorsNumber = 0

    i = 0
    for filename in ddd_a1.keys():
        for term in ddd_a1[filename].keys():
            if ddd_a1[filename][term]['cat'] == "Habitat":

                if term not in dl_terms.values():
                    id = "a1-t"+ str(i)
                    i+=1


                    l_words = nltk.tokenize.word_tokenize(term)
                    l_unicodeWords = list()
                    for word in l_words:
                        try:
                            unicodeWord = unicode(word)
                            l_unicodeWords.append(unicodeWord.lower())
                        except:
                            errorsNumber += 1

                    dl_terms[id] = l_unicodeWords


                    ddd_a1[filename][term]["termID"] = id

    return dl_terms, ddd_a1, errorsNumber



######################################################################################################################
# Pour remplir des fichiers A2 à partir d'associations données :
######################################################################################################################
def setFoundConcept(termID, lt_predictions):
    foundConcept = "<OBT:000000: bacteria habitat>"
    for pred in lt_predictions:
        if termID == pred[1]:
            foundConcept = pred[2]

    return foundConcept


def getIdConcept(concept):
    m = re.search(r"OBT:([0-9]*)", concept) # Use if direct parsin of A1/A2 file for training: m = re.search(r"<OBT:([0-9]*): (.*)>", concept)
    if m is not None:
        return m.group(1)
    else:
        print "m.group(0) =", m.group(0)


def writeA2File(a2filePath, dd_a1, lt_predictions):
    a2FileTest = open(a2filePath, "w")

    i = 1
    textOfFile = ""
    for term in dd_a1.keys():
        if dd_a1[term]["cat"] == "Habitat":

            l_Tid = dd_a1[term]["T"]

            termID = dd_a1[term]["termID"]
            foundConcept = setFoundConcept(termID, lt_predictions)

            idFoundConcept = getIdConcept(foundConcept)

            l_line = list()
            for T in l_Tid:
                N = "N" + str(i) + "\t"
                l_line.append(N + "OntoBiotope Annotation:" + T + " Referent:OBT:" + idFoundConcept + "\n")
                i += 1


        else:  # Bacteria: La référence doit être présente pour pouvoir être évaluée...

            l_Tid = dd_a1[term]["T"]

            l_line = list()
            for T in l_Tid:
                N = "N" + str(i) + "\t"
                l_line.append(N + "NCBI_Taxonomy Annotation:" + T + " Referent:2\n")  # 2 est l'id de "Bacteria"
                i += 1


        for line in l_line:
            textOfFile += line

    a2FileTest.write(textOfFile)
    a2FileTest.close()


def writeA2(a2Path, lt_predictions, ddd_a1):

    for filename in ddd_a1.keys():
        a2Filename = a2Path + "/" + filename + ".a2"
        writeA2File(a2Filename, ddd_a1[filename], lt_predictions)



#######################################################################################################
# Tests
#######################################################################################################
if __name__ == '__main__':

    a1Path = "../DEMO/DATA/BB-cat_test"
    ddd_a1 = parseA1(a1Path)
    print("ddd_a1 : "+str(ddd_a1))
    print("len(ddd_a1.keys()) :"+str(len(ddd_a1.keys())))
    dl_terms, ddd_a1 = getTermsFromA1(ddd_a1)


