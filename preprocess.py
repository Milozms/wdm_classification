import pickle
import re
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import numpy as np
num_train = 6000
num_test = 600
with open('unigrams.pk', 'rb') as f_unigrams:
    unigrams = pickle.load(f_unigrams)
with open('bigrams.pk', 'rb') as f_bigrams:
    bigrams = pickle.load(f_bigrams)
with open('trigrams.pk', 'rb') as f_trigrams:
    trigrams = pickle.load(f_trigrams)
dim = len(unigrams)+len(bigrams)+len(trigrams)
train_x = lil_matrix((num_train, dim), dtype=np.int8)
with open('train.in', 'r') as f:
    line_count = 0
    for line in f.readlines():
        line_split = re.findall(r"\w+(?:[-']\w+)*|'|[-.(\"]+|\S\w*", line)
        if len(line_split) > 0:
            for uni in line_split:
                if uni in unigrams:
                    train_x[line_count, unigrams[uni]] = 1
            line_split.append('\E')
            line_split = ['\S'] + line_split
            for i in range(0, len(line_split) - 1):
                bi = (line_split[i], line_split[i+1])
                if bi in bigrams:
                    train_x[line_count, bigrams[bi]] = 1
            line_split.append('\E')
            line_split = ['\S'] + line_split
            for i in range(0, len(line_split) - 2):
                tri = (line_split[i], line_split[i+1], line_split[i+2])
                if tri in trigrams:
                    train_x[line_count, trigrams[tri]] = 1
        line_count+=1
train_x = csr_matrix(train_x)
with open('trainx.pk', 'wb') as f_train:
    pickle.dump(train_x, f_train)

test_x = lil_matrix((num_test, dim), dtype=np.int8)
with open('test.in', 'r') as f:
    line_count = 0
    for line in f.readlines():
        line_split = re.findall(r"\w+(?:[-']\w+)*|'|[-.(\"]+|\S\w*", line)
        if len(line_split) > 0:
            for uni in line_split:
                if uni in unigrams:
                    test_x[line_count, unigrams[uni]] = 1
            line_split.append('\E')
            line_split = ['\S'] + line_split
            for i in range(0, len(line_split) - 1):
                bi = (line_split[i], line_split[i+1])
                if bi in bigrams:
                    test_x[line_count, bigrams[bi]] = 1
            line_split.append('\E')
            line_split = ['\S'] + line_split
            for i in range(0, len(line_split) - 2):
                tri = (line_split[i], line_split[i+1], line_split[i+2])
                if tri in trigrams:
                    test_x[line_count, trigrams[tri]] = 1
        line_count+=1
test_x = csr_matrix(test_x)
with open('testx.pk', 'wb') as f_test:
    pickle.dump(test_x, f_test)