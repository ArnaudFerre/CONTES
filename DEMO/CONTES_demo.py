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

import json, numpy, os

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

# Load an ontology for your task.
# (If you had many with your own task, you can use Protégé software to merge them in one)
from utils import onto
ontobiotiope = onto.loadOnto("DATA/OntoBiotope_BioNLP-ST-2016.obo")

# Building of concept embeddings and training:
from module_train import main_train
regMat, VSO, l_unknownTokens = main_train.train(word_vectors, dl_trainingTerms, attributions, ontobiotiope, factor=1.0)
print("Unknown tokens (possibly tokens from labels of the ontology): "+str(l_unknownTokens))


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

# Prediction with precedent training model (see STEP 2):
from module_predictor import main_predictor
lt_predictions, l_unknownTokens = main_predictor.predictor(word_vectors, dl_terms, VSO, regMat, 'cosine')
print("Unknown tokens in new mentions compare to W2V tokens: "+str(l_unknownTokens))

#################################################
# STEP 4: Formatting for evaluation
#################################################

# Create and write predictions in A2 files, put into the directory <<a2Path>>:
from utils import BioNLP_Format
a2Path = "../DEMO/DATA/predictedData"
BioNLP_Format.writeA2(a2Path, lt_predictions, ddd_a1)

###
# Find evaluation on: http://bibliome.jouy.inra.fr/demo/BioNLP-ST-2016-Evaluation/index.html
# Select BB-Cat Task
# Compress the prediction (A2 files) in a zip file and upload it.
# See the "Habitat only" precision.
###