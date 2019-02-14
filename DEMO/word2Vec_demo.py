#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: Processing of segmented corpus for Word2Vec (cerntaily not optimize)

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
import json, os

#######################################################################################################
# Run
#######################################################################################################
if __name__ == '__main__':

    # Loading a segmented corpus (list(list())):
    from module_word2vec import main_word2vec
    w2v = main_word2vec.Word2Vec()
    segmentedCorpusPath = "DATA/W2V_ExternalCorpus/"
    segmentedCorpusName = "filteredCorpus.json"
    segmentedExternalCorpusFile = open(segmentedCorpusPath + segmentedCorpusName, 'r')
    ll_segmentedExternalCorpus = json.load(segmentedExternalCorpusFile)

    # Lowercase and filter (remove stop-words) the corpus (move lowerAndFilter to True if not)
    lowerAndFilter = False #if already done
    if lowerAndFilter == True:
        # Filtering (if not done): remove stopwords and lowercase
        filtredCorpus = list()
        stopWords = set(stopwords.words('english'))
        for sentence in ll_segmentedExternalCorpus:
            l_filteredWords = list()
            for word in sentence:
                lowWord = word.lower()
                if lowWord not in stopWords:
                    l_filteredWords.append(unicode(lowWord))
            filtredCorpus.append(l_filteredWords)

        # Save the filtered corpus:
        filtredCorpusFile = open("DATA/W2V_ExternalCorpus/filteredCorpus.json", "w")
        json.dump(filtredCorpus, filtredCorpusFile)

    # Word2Vec
    w2v.corpus = ll_segmentedExternalCorpus
    minCountValue = 0
    vectSizeValue = 1000
    numIterationValue = 50
    w2v.buildVector(workerNum=8, minCount=minCountValue, vectSize=vectSizeValue, skipGram=True, numIteration=25)
    w2v.vst_model.save("DATA/wordEmbeddings/VST_count" + str(minCountValue) + "_size" + str(vectSizeValue) + "_iter" + str(numIterationValue) + ".model")

    # If you wan to explore your embeddings, you case use this:
    #print(w2v.most_similar(positive=['woman', 'king'], negative=['man']))
