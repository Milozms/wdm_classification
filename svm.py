import numpy as np
from scipy.sparse import csr_matrix
import pickle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
with open('trainx.pk', 'rb') as f_train:
    train_x = pickle.load(f_train)
with open('testx.pk', 'rb') as f_test:
    test_x = pickle.load(f_test)

train_y = []
test_y = []
with open('train.out', 'r') as f:
    for line in f.readlines():
        if len(line)>0 and line[0] != '\n':
            train_y.append(np.int8(int(line[0])))
train_y = np.array(train_y)
with open('test.out', 'r') as f:
    for line in f.readlines():
        if len(line)>0 and line[0] != '\n':
            test_y.append(np.int8(int(line[0])))
test_y = np.array(test_y)
print(test_y)
#clf = SVC(probability=True, verbose=True)
clf = LinearSVC()
clf.fit(train_x, train_y)
#print(clf.predict_proba(test_x))
#print(clf.get_params())
print(clf.predict(test_x))
print(clf.score(test_x, test_y))