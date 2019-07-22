# CONTES
CONcept-TErm System method to normalize multi-word terms with concepts from a domain-specific ontology (See [paper](http://www.aclweb.org/anthology/W17-2312)).

The system is based on |gensim| |sklearn|

## Intallation
1. Get CONTES from github

```
$ git clone https://github.com/ArnaudFerre/CONTES.git

$ cd CONTES
```

2. Create the Virtual Env, [You need anaconda to be installed](https://conda.io/en/latest/miniconda.html)

```
$ conda env create -f contes-env.yml
```

3. Activate the Virtual Env

```
$ source activate contesenv
```

4. Tests

```
$ python module_word2vec/main_word2vec.py --help

$ python module_train/main_train.py --help

$ python module_predictor/main_predictor.py --help
```


## Usage
* Calculate word embeddings

```
$ python module_word2vec/main_word2vec.py \
--json word-vectors.json \
--min-count 0 \
--vector-size 100 \
--window-size 2 < test/DATA/W2V_ExternalCorpus/corpus.txt
```

* Train a Contes model

```
$ python module_train/main_train.py  \
--word-vectors test/DATA/wordEmbeddings/VST_count0_size100_iter50.model \
--terms test/DATA/trainingData/terms_train.json \
--attributions test/DATA/trainingData/attributions_train.json \
--regression-matrix test/DATA/learnedHyperparameters/model.sav \
--ontology test/DATA/OntoBiotope_BioNLP-ST-2016.obo
```

* Predict from a Contes Model

```
$ python module_predictor/main_predictor.py \
--word-vectors test/DATA/wordEmbeddings/VST_count0_size100_iter50.model \
--terms test/DATA/trainingData/terms_train.json \
--regression-matrix test/DATA/learnedHyperparameters/model.sav \
--ontology test/DATA/OntoBiotope_BioNLP-ST-2016.obo
--output test/DATA/predictedData/output.json
``` 
