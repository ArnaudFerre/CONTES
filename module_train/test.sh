PYTHONPATH=/home/rbossy/code/CONTES python main_train.py --word-vectors test/word-vectors.json --terms test/terms.json --attributions test/attributions.json --ontology test/OntoBiotope_BioNLP-ST-2016.obo --ontology-vector ontology-vector.json --regression-matrix regression-matrix.json
diff -q ontology-vector.json test/ontology-vector.json
diff -q regression-matrix.json test/regression-matrix.json
