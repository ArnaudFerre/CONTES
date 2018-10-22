./main_word2vec.py --json output.json --txt output.txt --bin output.bin --vector-size 2 input.txt
diff -q output.txt test/output.txt
diff -q output.json test/output.json
diff -q output.bin test/output.bin
