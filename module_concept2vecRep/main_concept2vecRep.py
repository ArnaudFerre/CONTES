#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from sklearn import linear_model
from sklearn.externals import joblib
import numpy
import gensim

import os
from os.path import dirname, exists, abspath
from os import makedirs

from sys import stderr, stdin, path
from optparse import OptionParser
import json
import gzip

path.insert(0, os.path.abspath(".."))
from utils import word2term, onto

#######################################################################################################
# Functions
#######################################################################################################

def concept2vecRep(ontopath, mode="Ancestry", factor=None):

    # Load an ontology for your task.
    # (If you had many with your own task, you can use Protégé software to merge them in one with a unique root)
    myOnto = onto.loadOnto(ontopath)

    if mode == "Ancestry":
        if factor == None:
            factor = 1.0

        vso = onto.ontoToVec(myOnto, factor)

    return vso



class Concept2vecRep(OptionParser):
    def __init__(self):
        OptionParser.__init__(self, usage='usage: %prog [options]')

        # Input:
        self.add_option('--ontology', action='store', type='string', dest='ontology', help='path to ontology file in OBO format')

        # Parameters
        self.add_option('--mode', action='append', type='string', dest='mode', default='Ancestry', help='mode of ontological representations process')
        self.add_option('--factor', action='append', type='float', dest='factor', default=[], help='parent concept weight factor (default: 1.0)')

        # Output:
        self.add_option('--ontology-vector', action='store', type='string', dest='ontology_vector', help='path to the ontology vector file')

    def run(self):
        options, args = self.parse_args()
        if len(args) > 0:
            raise Exception('stray arguments: ' + ' '.join(args))
        if options.ontology is None:
            raise Exception('missing --ontology')
        if options.mode == "Ancestry" and isinstance(options.factor, float)==False:
            raise Exception('invalid factor for Ancestry mode (float)')

        stderr.write('calculating representations of ontology: %s\n' % options.ontology)
        stderr.flush()
        VSO = concept2vecRep(options.ontology, mode="Ancestry", factor=None)

        serializable = dict((k, list(v)) for k, v in VSO.iteritems())
        stderr.write('writing ontology vector: %s\n' % options.ontology_vector)
        stderr.flush()
        f = open(options.ontology_vector, 'w')
        json.dump(serializable, f)
        f.close()



if __name__ == '__main__':
    Concept2vecRep().run()
