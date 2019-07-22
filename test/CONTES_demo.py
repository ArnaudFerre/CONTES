#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding: utf-8

"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description:

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json, numpy, os, sys
sys.path.insert(0, os.path.abspath(".."))


#################################################
# STEP 1: Training & Ontological embedding
#################################################

# Load an existing W2V model (Gensim format):
from gensim.models import Word2Vec
modelPath = "DATA/wordEmbeddings/VST_count0_size100_iter50.model" # the provided models are really small models, just to test execution
filename, file_extension = os.path.splitext(modelPath)
print("Loading word embeddings...")
if file_extension == ".model":
    model = Word2Vec.load(modelPath)
    word_vectors = dict((k, list(numpy.float_(npf32) for npf32 in model.wv[k])) for k in model.wv.vocab.keys()) # To improve, to take directly a binary model from Gensim.
    del model
elif file_extension == ".json":
    VSTjsonFile = open(modelPath, 'r')
    word_vectors = json.load(VSTjsonFile)
print("Word embeddings loaded.\n")

# Automatic load of training data:
mentionsFilePath = "DATA/trainingData/terms_trainObo.json"
attributionsFilePath = "DATA/trainingData/attributions_trainObo.json"
extractedMentionsFile = open(mentionsFilePath , 'r')
dl_trainingTerms = json.load(extractedMentionsFile)
attributionsFile = open(attributionsFilePath, 'r')
attributions = json.load(attributionsFile)

######

# Calculate vector representations of concepts:
from module_concept2vecRep import main_concept2vecRep
ontoPath = "DATA/OntoBiotope_BioNLP-ST-2016.obo"
VSO = main_concept2vecRep.concept2vecRep(ontoPath, mode="Ancestry", factor=1.0)

# Write VSO in a JSON file:
print("Writing of VSO...")
VSO_path = "DATA/VSO_OntoBiotope_BioNLP-ST-2016.json"
serializable = dict((k, list(v)) for k, v in VSO.iteritems())
f = open(VSO_path, 'w')
json.dump(serializable, f)
f.close()
print("VSO has been written.\n")

######

# Calculate vector representations for expressions (possibly multiwords):
from module_wordVec2ExpVec import main_wordVec2expVec
VST, l_unknownToken = main_wordVec2expVec.wordVST2TermVST(word_vectors, dl_trainingTerms)
print l_unknownToken

######

# Building of concept embeddings and training:
from module_train import main_train
print("Loading VSO...")
VSO_path = "DATA/VSO_OntoBiotope_BioNLP-ST-2016.json"
VSO_file = open(VSO_path, "r")
VSO = json.load(VSO_file)
print("VSO loaded.\n")
regMat = main_train.train(VST, dl_trainingTerms, attributions, VSO)

#################################################
# STEP 3: Prediction
#################################################

# Parsing of A1 files:
from utils import BioNLP_Format
a1Path = "DATA/BB-cat_dev" #"DATA/BB-cat_test"
ddd_a1 = BioNLP_Format.parseA1(a1Path)

# Generate unique mention IDs:
dl_terms, ddd_a1, errorsNumber = BioNLP_Format.getTermsFromA1(ddd_a1)
print("Number of unreadable tokens: "+str(errorsNumber))

######

# Calculate vector representations for expressions (possibly multiwords):
from module_wordVec2ExpVec import main_wordVec2expVec
VST, l_unknownToken = main_wordVec2expVec.wordVST2TermVST(word_vectors, dl_terms)
print l_unknownToken

######

# Prediction with precedent training model (see STEP 2):
from module_predictor import main_predictor
lt_predictions = main_predictor.predictor(VST, dl_terms, VSO, regMat, 'cosine')

#################################################
# STEP 4: Formatting for evaluation
#################################################

# Create and write predictions in A2 files, put into the directory <<a2Path>>:
from utils import BioNLP_Format
a2Path = "../test/DATA/predictedData"
BioNLP_Format.writeA2(a2Path, lt_predictions, ddd_a1)

###
# Find evaluation on: http://bibliome.jouy.inra.fr/demo/BioNLP-ST-2016-Evaluation/index.html
# Select BB-Cat Task
# Compress the prediction (A2 files) in a zip file and upload it.
# See the "Habitat only" precision.
###