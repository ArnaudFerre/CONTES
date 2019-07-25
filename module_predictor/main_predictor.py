#!/usr/bin/env python3
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
from sklearn.externals import joblib
import sys, os
from optparse import OptionParser
import json
import gzip
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from numpy import asarray

sys.path.insert(0, os.path.abspath(".."))
from utils import word2term, onto

#######################################################################################################
# Functions
#######################################################################################################


def metric_internal(metric):
    if metric == 'cosine':
        return 'euclidean'
    if metric == 'cosine-brute':
        return 'cosine'
    return metric

def metric_norm(metric, concept_vectors):
    if metric == 'cosine':
        return normalize(concept_vectors)
    return concept_vectors

def metric_sim(metric, d, vecTerm, vecConcept):
    if metric == 'cosine':
        return 1 - cosine(vecTerm, vecConcept)
    if metric == 'cosine-brute':
        return 1 - d
    return 1 / d



class VSONN(NearestNeighbors):
    def __init__(self, vso, metric):
        NearestNeighbors.__init__(self, algorithm='auto', metric=metric_internal(metric))
        self.original_metric = metric
        self.vso = vso
        self.concepts = tuple(vso.keys())
        self.concept_vectors = list(vso.values())
        self.fit(metric_norm(metric, self.concept_vectors))

    def nearest_concept(self, vecTerm):
        r = self.kneighbors([vecTerm], 1, return_distance=True)
        #stderr.write('r = %s\n' % str(r))
        d = r[0][0][0]
        idx = r[1][0][0]
        return self.concepts[idx], metric_sim(self.original_metric, d, vecTerm, self.concept_vectors[idx])




def predictor(vstTerm, dl_terms, vso, transformationParam, metric, symbol='___'):
    """
    Description: From a calculated linear projection from the training module, applied it to predict a concept for each
        terms in parameters (dl_terms).
    :param vst_onlyTokens: An initial VST containing only tokens and associated vectors.
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :param vso: A VSO (dict() -> {"id" : [vector], ...}
    :param transformationParam: LinearRegression object from Sklearn. Use the one calculated by the training module.
    :param symbol: Symbol delimiting the different token in a multi-words term.
    :return: A list of tuples containing : ("term form", "term id", "predicted concept id").
    """
    lt_predictions = list()

    result = dict()

    vsoTerms = dict()
    vsoNN = VSONN(vso, metric)
    for id_term in dl_terms.keys():
        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        x = vstTerm[termForm].reshape(1, -1)
        vsoTerms[termForm] = transformationParam.predict(x)[0]
        result[termForm] = vsoNN.nearest_concept(vsoTerms[termForm])

    for id_term in dl_terms.keys():
        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        cat, sim = result[termForm]
        prediction = (termForm, id_term, cat, sim)
        lt_predictions.append(prediction)

    return lt_predictions


def loadJSON(filename):
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        f = open(filename, encoding='utf-8')
    result = json.load(f)
    f.close()
    return result;


class Predictor(OptionParser):
    def __init__(self):
        OptionParser.__init__(self, usage='usage: %prog [options]')
        self.add_option('--ontology-vector', action='store', type='string', dest='vsoPath', help='path to the ontology vector file')
        self.add_option('--terms', action='append', type='string', dest='terms', help='path to terms file in JSON format (map: id -> array of tokens)')
        self.add_option('--factor', action='append', type='float', dest='factors', default=[], help='parent concept weight factor (default: 1.0)')
        self.add_option('--regression-matrix', action='append', type='string', dest='regression_matrix', help='path to the regression matrix file as produced by the training module')
        self.add_option('--vst', action='append', type='string', dest='vstPath', help='path to terms vectors file in JSON format (map: token1___token2 -> array of floats)')
        self.add_option('--metric', action='store', type='string', dest='metric', default='cosine', help='distance metric to use (default: %default)')
        self.add_option('--output', action='append', type='string', dest='output', help='file where to write predictions')


    def run(self):
        options, args = self.parse_args()
        if len(args) > 0:
            raise Exception('stray arguments: ' + ' '.join(args))
        if options.ontology is None:
            raise Exception('missing --ontology')
        if not(options.terms):
            raise Exception('missing --terms')
        if not(options.regression_matrix):
            raise Exception('missing --regression-matrix')
        if not(options.output):
            raise Exception('missing --output')
        if len(options.terms) != len(options.regression_matrix):
            raise Exception('there must be the same number of --terms and --regression-matrix')
        if len(options.terms) != len(options.output):
            raise Exception('there must be the same number of --terms and --output')
        if len(options.factors) > len(options.terms):
            raise Exception('there must be at least as many --terms as --factor')
        if len(options.factors) < len(options.terms):
            n = len(options.terms) - len(options.factors)
            sys.stderr.write('defaulting %d factors to 1.0\n' % n)
            sys.stderr.flush()
            options.factors.extend([1.0]*n)
        if options.vsoPath is None:
            raise Exception('missing --ontology-vector')

        for terms_i, regression_matrix_i, output_i, factor_i in zip(options.terms, options.regression_matrix, options.output, options.factors):
            sys.stderr.write('loading ontology-vector: %s\n' % options.vsoPath)
            sys.stderr.flush()
            vso = json.load(options.vsoPath)

            sys.stderr.write('loading terms: %s\n' % terms_i)
            sys.stderr.flush()
            terms = loadJSON(terms_i)
            sys.stderr.write('loading regression matrix: %s\n' % regression_matrix_i)
            sys.stderr.flush()
            regression_matrix = joblib.load(regression_matrix_i)
            sys.stderr.write('loading expressions embeddings: %s\n' % options.vsoPath)
            sys.stderr.flush()
            vstTerm = loadJSON(options.vsoPath)

            sys.stderr.write('predicting\n')
            sys.stderr.flush()
            prediction, _ = predictor(vstTerm, terms, vso, regression_matrix, options.metric)

            sys.stderr.write('writing predictions: %s\n' % output_i)
            sys.stderr.flush()
            f = open(output_i, 'w')
            for _, term_id, concept_id, similarity in prediction:
                f.write('%s\t%s\t%f\n' % (term_id, concept_id, similarity))
            f.close()



if __name__ == '__main__':

    # Path to test data:
    mentionsFilePath = "../test/DATA/trainingData/terms_trainObo.json"
    vst_termsPath = "../test/DATA/expressionEmbeddings/vstTerm_trainObo.json"
    SSO_path = "../test/DATA/VSO_OntoBiotope_BioNLP-ST-2016.json"
    regmatPath = "../test/DATA/learnedHyperparameters/model.sav"
    predictionPath = "../test/DATA/predictedData/prediction_trainOboONtrainObo.json"

    # load input data:
    print("\nLoading data...")
    extractedMentionsFile = open(mentionsFilePath, 'r')
    dl_testedTerms = json.load(extractedMentionsFile)
    extractedMentionsFile.close()
    vstTermsFile = open(vst_termsPath, 'r')
    vst_terms = json.load(vstTermsFile)
    for term in vst_terms.keys():
        vst_terms[term] = asarray(vst_terms[term])
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
    lt_predictions = predictor(vst_terms, dl_testedTerms, SSO, transformationParam, metric, symbol='___')
    print("Prediction done.\n")

    print("Saving prediction results...")
    f = open(predictionPath, 'w')
    json.dump(lt_predictions, f)
    f.close()
    print("Saving done.")


    #Predictor().run()
