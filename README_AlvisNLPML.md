# CONTES/HONOR

This installation guide enable the use of two methods of normalization with concepts from a domain-specific ontology: CONTES and HONOR through the corpus processing engine [AlvisNLP/ML](https://bibliome.github.io/alvisnlp/).
For more details on the methods, see related papers:
- [CONTES](http://www.aclweb.org/anthology/W17-2312): Representation of complex terms in a vector space structured by an ontology for a normalization task. Ferré, A., Zweigenbaum, P., & Nédellec, C. In 2017 ACL workshop BioNLP.
- [HONOR](https://www.aclweb.org/anthology/L18-1543): Combining rule-based and embedding-based approaches to normalize textual entities with an ontology. Ferré, A., Deléger, L., Zweigenbaum, P., & Nédellec, C. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC-2018).

## Prerequisites

* Python 3.x (tested with Anaconda and Python 3.7)
* [Gensim](https://radimrehurek.com/gensim/install.html)
* [Scikit-Learn](https://scikit-learn.org/stable/install.html)

Comment: Using [Anaconda](https://www.anaconda.com/distribution/) with Python 3.x will install most of the usual libs needed by these prerequisited libs and enable some other practical things (like virtual environment). We recommand their use.


## Intallation
1. Get AlvisNLP/ML from GitHub:
```
$ git clone https://github.com/Bibliome/alvisnlp.git
```

2. Follow the instructions from:
- For Linux OS: [https://github.com/Bibliome/alvisnlp/blob/master/README.md](https://github.com/Bibliome/alvisnlp/blob/master/README.md)
- For Win OS: [https://github.com/Bibliome/alvisnlp/blob/master/WINDOWS.md](https://github.com/Bibliome/alvisnlp/blob/master/WINDOWS.md)

Comments:
- Be sure that you are root/administrator user.
- (Optional) If you wish to apply some specific preprocessing to your corpus: get [TreeTagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) tool.
- You can jump the "Default command-line options" and "Web service" sections.
- On Windows, use '\\' instead of '/' in path.

3. In your base directory of AlvisNLP/ML, fill the CONTES (and TreeTagger if installed) modules in the `default-param-values.xml` file:
```
<module class="fr.inra.maiage.bibliome.alvisnlp.bibliomefactory.modules.treetagger.TreeTagger">
	<treeTaggerExecutable>**pathToYourTreeTaggerExecutable**</treeTaggerExecutable>
	<parFile>**pathToYourTreeTaggerDirectory**/lib/english-utf8.par</parFile>
	<inputCharset>UTF-8</inputCharset>
	<outputCharset>UTF-8</outputCharset>
</module>

<module class="fr.inra.maiage.bibliome.alvisnlp.bibliomefactory.modules.contes.Word2Vec">
	<python3Executable>**pathToPythonExe**</python3Executable>
	<contesDir>**pathToYourContesDirectory**</contesDir>
	<workers>**DefaultNumberOfWorkers**</workers>
</module>

<module class="fr.inra.maiage.bibliome.alvisnlp.bibliomefactory.modules.contes.ContesTrain">
	<contesDir>**pathToYourContesDirectory**</contesDir>
	<python3Executable>**pathToPythonExe**</python3Executable>
</module>

<module class="fr.inra.maiage.bibliome.alvisnlp.bibliomefactory.modules.contes.ContesPredict">
	<contesDir>**pathToYourContesDirectory**</contesDir>
	<python3Executable>**pathToPythonExe**</python3Executable>
</module>
```

4. Check your installation, for instance with the `word2vec.plan` in `alvisnlp/alvisnlp-test/contes`, on Linux:
```
$ ./../../../baseDirectoryAlvis/bin/alvisnlp.bat alvisnlp word2vec.plan -verbose -inputDir **yourAbsolutePathTo**/alvisnlp/alvisnlp-test/share -inputDir .
```
Or on Windows:
```
$ ..\..\..\baseDirectoryAlvis\bin\alvisnlp.bat alvisnlp word2vec.plan -verbose -inputDir **yourAbsolutePathTo**\alvisnlp\alvisnlp-test\share -inputDir .
```


## CONTES 

The CONTES method consists of two modules: [ContesTrain](https://bibliome.github.io/alvisnlp/reference/module/ContesTrain) and [ContesPredict](https://bibliome.github.io/alvisnlp/reference/module/ContesPredict). The both require word embeddings that can be generated with the [Word2Vec module](https://bibliome.github.io/alvisnlp/reference/module/Word2Vec) or another source. 

You can find template of CONTES/HONOR AlvisNLP/ML plans in `alvisnlp\alvisnlp-test\contes`. By default, these templates use the `read` plan which is in `alvisnlp\alvisnlp-test\share\BioNLP-ST-2016_BB-cat+ner`. This plan uses reader module [BioNLPSTReader](https://bibliome.github.io/alvisnlp/reference/module/BioNLPSTReader) to load the open data of the Bacterai Biotope normalization task of [BioNLP-ST 2016](http://2016.bionlp-st.org/tasks/bb2). They also use the OntoBiotope ontology of the task (2016 version), which is in `alvisnlp\alvisnlp-test\share\`. Check the [documentation](https://bibliome.github.io/alvisnlp/reference/Module-reference#readers) on reader modules to load other data. You also need the `normalization.plan` in `alvisnlp\alvisnlp-test\contes`, which operates some preprocessings on all data (stopwords filtering, lowercase, ...).

1. Training step: ContesTrain
The module use a set of word embeddings and an ontology. You can try this command line to use the plan:
```
$ ..\baseDirectoryAlvis\bin\alvisnlp.bat train.plan -verbose -inputDir . -inputDir **yourAbsolutePathTo**\alvisnlp\alvisnlp-test\share
```
The plan will create a sklearn file `regression.bin`, which contains the learned parameters of the training. Check [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) for more details on this file.

2. Prediction step: ContesPredict
The module use a set of word embeddings, an ontology and the a `regression.bin`. You can try this command line to use the plan:
```
$ ..\baseDirectoryAlvis\bin\alvisnlp.bat predict.plan -verbose -inputDir . -inputDir **yourAbsolutePathTo**\alvisnlp\alvisnlp-test\share
```

3. Evaluation
For the Bacteria Biotope task of BioNLP-ST 2016, you can evaluate [online](http://bibliome.jouy.inra.fr/demo/BioNLP-ST-2016-Evaluation/index.html) the performance of the method.
 

## HONOR 

Ongoing...

