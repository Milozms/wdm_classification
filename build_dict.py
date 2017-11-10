import re
import pickle
unigrams = {}
bigrams = {}
trigrams = {}
count_u = 0
count_b = 0
count_t = 0

with open('train.in', 'r') as f:
    for line in f.readlines():
        line_split = re.findall(r"\w+(?:[-']\w+)*|'|[-.(\"]+|\S\w*", line)
        for uni in line_split:
            if uni not in unigrams:
                unigrams[uni] = count_u
                count_u += 1
        line_split.append('\E')
        line_split = ['\S'] + line_split
        for i in range(0, len(line_split) - 1):
            bi = (line_split[i], line_split[i+1])
            if bi not in bigrams:
                bigrams[bi] = count_b
                count_b += 1
        line_split.append('\E')
        line_split = ['\S'] + line_split
        for i in range(0, len(line_split) - 2):
            tri = (line_split[i], line_split[i+1], line_split[i+2])
            if tri not in trigrams:
                trigrams[tri] = count_t
                count_t += 1
                
with open('unigrams.pk', 'wb') as f_unigrams:
    pickle.dump(unigrams, f_unigrams)
with open('bigrams.pk', 'wb') as f_bigrams:
    pickle.dump(bigrams, f_bigrams)
with open('trigrams.pk', 'wb') as f_trigrams:
    pickle.dump(trigrams, f_trigrams)