import pickle
import re
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import numpy as np
num_train = 6000
with open('unigrams.pk', 'rb') as f_unigrams:
    unigrams = pickle.load(f_unigrams)
with open('bigrams.pk', 'rb') as f_bigrams:
    bigrams = pickle.load(f_bigrams)
with open('trigrams.pk', 'rb') as f_trigrams:
    trigrams = pickle.load(f_trigrams)
dim = len(unigrams)+len(bigrams)+len(trigrams)
#train_x = csr_matrix((num_train, dim), dtype=np.int8)
train_x = lil_matrix((num_train, dim), dtype=np.int8)
#row = []
#col = []
#data = []
with open('train.in', 'r') as f:
    line_count = 0
    for line in f.readlines():
        line_split = re.findall(r"\w+(?:[-']\w+)*|'|[-.(\"]+|\S\w*", line)
        if len(line_split) > 0 and line_count<4:
            for uni in line_split:
                if uni in unigrams:
                    #row.append(line_count)
                    #col.append(unigrams[uni])
                    #data.append(1)
                    train_x[line_count, unigrams[uni]] = 1
            line_split.append('\E')
            line_split = ['\S'] + line_split
            for i in range(0, len(line_split) - 1):
                bi = (line_split[i], line_split[i+1])
                if bi in bigrams:
                    #row.append(line_count)
                    #col.append(bigrams[bi])
                    #data.append(1)
                    train_x[line_count, bigrams[bi]] = 1
            line_split.append('\E')
            line_split = ['\S'] + line_split
            for i in range(0, len(line_split) - 2):
                tri = (line_split[i], line_split[i+1], line_split[i+2])
                if tri in trigrams:
                    #row.append(line_count)
                    #col.append(trigrams[tri])
                    #data.append(1)
                    train_x[line_count, trigrams[tri]] = 1
        line_count+=1
#print(data)
#print(row)
#print(col)
#train_x = csr_matrix((data, (row, col)), shape=(line_count, dim), dtype=np.int8)
train_x = csr_matrix(train_x)
for i in range(0, 100000):
    print(train_x.toarray()[0][i])
for i in range(0, 100000):
    if train_x.toarray()[100][i] > 0:
        print(i)
for i in range(0, 600):
    print(np.sum(test_x.toarray()[i]))