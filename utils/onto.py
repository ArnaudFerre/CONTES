#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: Blablabla.
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
import pronto
import numpy


#######################################################################################################
# Functions
#######################################################################################################

def loadOnto(ontoPath):
    """
    Description: Load an ontology object from a specified path.
    :param ontoPath: path of the ontology
    :return: pronto object representing an ontology
    NB: Even if it should accept OWL file, only OBO seems work.
    """
    onto = pronto.Ontology(ontoPath)
    return onto



def ancestor_level(concept, level, levelmap):
    levelmap[concept.id] = level
    for parent in concept.parents:
        ancestor_level(parent, level+1, levelmap)
    return levelmap


def ontoToVec(onto, factor=1.0):
    """
    Description: Create a vector space of the ontology. It uses hierarchical information to do this.
    :param onto: A Pronto object representing an ontology.
    :return: A VSO, that is a dictionary with id of concept (<XXX_xxxxxxxx: Label>) as keys and a numpy vector in value.
    """
    vso = dict()

    size = len(onto)
    d_assoDim = dict()

    for i, concept in enumerate(onto):
        id_concept = concept.id
        vso[id_concept] = numpy.zeros(size)
        d_assoDim[id_concept] = i
        #vso[id_concept][d_assoDim[id_concept]] = 1

    for concept in onto:
        levelmap = ancestor_level(concept, 0, {})
        id_concept = concept.id
        for id_ancestor, level in levelmap.items():
            vso[id_concept][d_assoDim[id_ancestor]] = factor**level
        # for parent in concept.rparents(-1, True):
        #     id_parent = parent.id
        #     vso[id_concept][d_assoDim[id_parent]] = 1

    return vso

