# CONTES
CONcept-TErm System method to normalize multi-word terms with concepts from a domain-specific ontology (See: [paper](http://www.aclweb.org/anthology/W17-2312)).

The system is based on |gensim| |sklearn|

## Intallation
1. Get CONTES from github

`
git clone https://github.com/ArnaudFerre/CONTES.git
`

`
cd CONTES
`

2. Create the Virtual Env

`
conda env create -f contes-env.yml
`

3. Activate the Virtual Env

`
source activate contesenv
`

4. Test

`
python module_word2vec/main_word2vec.py
`

`
python module_train/main_train.py
`

`
python module_predictor/main_predictor.py
`


## Usage
