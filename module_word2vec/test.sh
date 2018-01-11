./main_word2vec.py --json output.json --txt output.txt --vector-size 2 input.txt
diff -q output.txt test/output.txt
diff -q output.json test/output.json
