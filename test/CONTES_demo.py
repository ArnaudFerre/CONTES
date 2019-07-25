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
from sklearn.externals import joblib

sys.path.insert(0, os.path.abspath(".."))
from gensim.models import Word2Vec
from utils import BioNLP_Format
from module_wordVec2ExpVec import main_wordVec2expVec
from module_concept2vecRep import main_concept2vecRep
from module_train import main_train
from module_predictor import main_predictor


#################################################
# STEP 1: Expression embeddings processing
#################################################


# Load an existing W2V model (Gensim format):
print("\nLoading word embeddings...")
modelPath = "DATA/wordEmbeddings/VST_count0_size200_iter10.model" #"DATA/wordEmbeddings/VST_count0_size100_iter50.model" # the provided models are really small models, just to test execution
filename, file_extension = os.path.splitext(modelPath)
if file_extension == ".model":
    model = Word2Vec.load(modelPath)
    word_vectors = dict((k, list(numpy.float_(npf32) for npf32 in model.wv[k])) for k in model.wv.vocab.keys()) # To improve, take directly a binary model from Gensim.
    del model
elif file_extension == ".json":
    VSTjsonFile = open(modelPath, 'r')
    word_vectors = json.load(VSTjsonFile)
    VSTjsonFile.close()
print("Word embeddings loaded.\n")

# Load expressions from training dataset:
print("Loading expressions from training dataset...")
trainMentionsFilePath = "DATA/trainingData/terms_trainObo.json"
extractedTrainMentionsFile = open(trainMentionsFilePath, 'r')
dl_trainingTerms = json.load(extractedTrainMentionsFile)
extractedTrainMentionsFile.close()
print("Expressions from training dataset loaded.\n")

# Calculate vector representations for train expressions (possibly multiwords):
print("Calculating vector representations for train expressions...")
vst_trainTerms, l_unknownToken = main_wordVec2expVec.wordVST2TermVST(word_vectors, dl_trainingTerms)
print("Number of out-of-vocabulary words in training expressions: " + str(len(l_unknownToken)))
for term in vst_trainTerms.keys():
    vst_trainTerms[term] = list(vst_trainTerms[term])
vst_termsTrainPath = "../test/DATA/expressionEmbeddings/vstTerm_trainObo.json"
vstTrainTermsFile = open(vst_termsTrainPath, 'w')
json.dump(vst_trainTerms, vstTrainTermsFile)
vstTrainTermsFile.close()
print("Done.\n")

# Parsing of A1 files for test expressions:
print("Parsing of test dataset...")
a1Path = "DATA/BB-cat_dev" #"DATA/BB-cat_test"
ddd_a1 = BioNLP_Format.parseA1(a1Path)
dl_testTerms, ddd_a1, errorsNumber = BioNLP_Format.getTermsFromA1(ddd_a1) #Generate unique mention IDs
print("Number of unreadable tokens: "+ str(errorsNumber))
print("Parsing done.\n")

# Saving expressions from test dataset and A1 files info:
print("Writing of expressions from test dataset...")
testMentionsFilePath = "DATA/trainingData/terms_dev.json"
f = open(testMentionsFilePath, 'w')
json.dump(dl_testTerms, f)
f.close()
A1_file_info_path = "DATA/trainingData/A1_dev.json"
f = open(A1_file_info_path, 'w')
json.dump(ddd_a1, f)
f.close()
print("Expressions and A1 files info has been saved.\n")

# Calculate vector representations for test expressions (possibly multiwords):
print("Calculating vector representations for test expressions...")
vst_testTerms, l_unknownToken = main_wordVec2expVec.wordVST2TermVST(word_vectors, dl_testTerms)
print("Number of out-of-vocabulary words in test expressions: " + str(len(l_unknownToken)))
for term in vst_testTerms.keys():
    vst_testTerms[term] = list(vst_testTerms[term])
vst_termsTestPath = "../test/DATA/expressionEmbeddings/vstTerm_dev.json"
vstTestTermsFile = open(vst_termsTestPath, 'w')
json.dump(vst_testTerms, vstTestTermsFile)
vstTestTermsFile.close()
vstTrainTermsFile.close()
print("Done.\n")

# Erase of variables:
print("In memory variables erasing...")
del word_vectors
del dl_trainingTerms
del ddd_a1
del dl_testTerms
del vst_trainTerms
del vst_testTerms
print("Erasing done.\n\n\n")




#################################################
# STEP 2: Concept embeddings processing
#################################################


# Calculate vector representations of concepts:
print("Calculating VSO...")
ontoPath = "DATA/OntoBiotope_BioNLP-ST-2016.obo"
VSO = main_concept2vecRep.concept2vecRep(ontoPath, mode="Ancestry", factor=1.0)
print("Done.\n")

# Write VSO in a JSON file:
print("Writing of VSO...")
VSO_path = "DATA/VSO_OntoBiotope_BioNLP-ST-2016.json"
serializable = dict((k, list(v)) for k, v in VSO.iteritems())
f = open(VSO_path, 'w')
json.dump(serializable, f)
f.close()
print("VSO has been saved.\n")

# Erase of variables:
print("In memory variables erasing...")
del VSO
del serializable
print("Erasing done.\n\n\n")


#################################################
# STEP 3: Training
#################################################


# Reloading od VSO:
print("Loading VSO for training...")
VSO_path = "DATA/VSO_OntoBiotope_BioNLP-ST-2016.json"
VSO_file = open(VSO_path, "r")
VSO = json.load(VSO_file)
print("VSO loaded.\n")

# Load training data:
print("Loading training data...")
mentionsFilePath = "DATA/trainingData/terms_trainObo.json"
attributionsFilePath = "DATA/trainingData/attributions_trainObo.json"
vst_termsTrainPath = "../test/DATA/expressionEmbeddings/vstTerm_trainObo.json"
extractedMentionsFile = open(mentionsFilePath, 'r')
dl_trainingTerms = json.load(extractedMentionsFile)
attributionsFile = open(attributionsFilePath, 'r')
attributions = json.load(attributionsFile)
vstTermsFile = open(vst_termsTrainPath, 'r')
vst_trainTerms = json.load(vstTermsFile)
print("Training data loaded.\n")

# Training:
print("Training of the CONTES method...")
regMat = main_train.train(vst_trainTerms, dl_trainingTerms, attributions, VSO)
print("Training done.\n")

print("Saving learned hyperparameters...")
regmatPath = "DATA/learnedHyperparameters/trainObo_model.sav"
joblib.dump(regMat, regmatPath)
print("Saving done.\n")

# Erase of variables:
print("In memory variables erasing...")
del VSO
del dl_trainingTerms
del vst_trainTerms
del regMat
print("Erasing done.\n\n\n")



#################################################
# STEP 4: Prediction
#################################################


# load input data:
print("Loading data...")
testMentionsFilePath = "DATA/trainingData/terms_dev.json"
vst_termsTestPath = "../test/DATA/expressionEmbeddings/vstTerm_dev.json"
SSO_path = "../test/DATA/VSO_OntoBiotope_BioNLP-ST-2016.json"
regmatPath = "DATA/learnedHyperparameters/trainObo_model.sav"
predictionPath = "../test/DATA/predictedData/prediction_trainObo_on_dev.json"
extractedMentionsFile = open(testMentionsFilePath, 'r')
dl_testedTerms = json.load(extractedMentionsFile)
extractedMentionsFile.close()
vstTermsFile = open(vst_termsTestPath, 'r')
vst_terms = json.load(vstTermsFile)
for term in vst_terms.keys():
    vst_terms[term] = numpy.asarray(vst_terms[term])
vstTermsFile.close()
transformationParam = joblib.load(regmatPath)
print("Data loaded.\n")

# Building of concept embeddings and training:
print("Loading Semantic Space of the Ontology (SSO)...")
SSO_file = open(SSO_path, "r")
SSO = json.load(SSO_file)
print("SSO loaded.\n")

print("Prediction...")
metric = "cosine"
lt_predictions = main_predictor.predictor(vst_terms, dl_testedTerms, SSO, transformationParam, metric, symbol='___')
print("Prediction done.\n")

print("Saving prediction results...")
f = open(predictionPath, 'w')
json.dump(lt_predictions, f)
f.close()
print("Saving done.\n")

# Erase of variables:
print("In memory variables erasing...")
del dl_testedTerms
del vst_terms
del SSO
del transformationParam
del lt_predictions
print("Erasing done.\n\n\n")



#################################################
# STEP 4: Formatting for evaluation
#################################################


# Load predictions:
print("Loading predictions...")
predictionPath = "../test/DATA/predictedData/prediction_trainObo_on_dev.json"
prediction_file = open(predictionPath, "r")
lt_predictions = json.load(prediction_file)
A1_file_info_path = "DATA/trainingData/A1_dev.json"
f = open(A1_file_info_path, "r")
ddd_a1 = json.load(f)
f.close()
print("Predictions loaded.\n")

# Create and write predictions in A2 files, put into the directory <<a2Path>>:
print("Formatting result in A2 format...")
a2Path = "../test/DATA/predictedData"
BioNLP_Format.writeA2(a2Path, lt_predictions, ddd_a1)
print("Formatting done.")



###
# Find evaluation on: http://bibliome.jouy.inra.fr/demo/BioNLP-ST-2016-Evaluation/index.html
# Select BB-Cat Task
# Compress the prediction (A2 files) in a zip file and upload it.
# See the "Habitat only" precision.
###