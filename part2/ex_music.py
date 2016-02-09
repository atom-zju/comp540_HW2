import numpy as np
import music_utils
from one_vs_all import one_vs_allLogisticRegressor
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


import numpy as np
import matplotlib.pyplot as plt


def plot_lambda_selection(reg_vec,error_train,error_val):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axes = plt.gca()
    axes.set_ylim([0,1.2])
    plt.plot(reg_vec,error_train,'b-',reg_vec,error_val,'g-')
    plt.title('Variation in training/validation error with lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Training/Validation Accuracy')
    plt.legend(["Training Accuracy","Validation Accuracy"])
    ax.set_xscale('log')

# some global constants

MUSIC_DIR = "music/"
genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

# select the CEPS or FFT representation

X,y = music_utils.read_ceps(genres,MUSIC_DIR)
#X,y = music_utils.read_fft(genres,MUSIC_DIR)

# select a regularization parameter

reg = 1.0

# create a 1-vs-all classifier

ova_logreg = one_vs_allLogisticRegressor(np.arange(10))

#  divide X into train and test sets

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# train the K classifiers in 1-vs-all mode

ova_logreg.train(X_train,y_train,reg,'l2')

# predict on the set aside test set

ypred = ova_logreg.predict(X_test)
print classification_report(y_test, ypred)
print "The model accuracy is : "+str(accuracy_score(y_test, ypred))
print confusion_matrix(y_test,ypred)

reg_vec = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50, 100, 300, 500, 1000]
error_train = np.zeros((len(reg_vec),))
error_val = np.zeros((len(reg_vec),))
idx = 0
for reg in reg_vec:
    ova_logreg.train(X_train,y_train,reg,'l2')
    y_train_pred = ova_logreg.predict(X_train)
    ypred = ova_logreg.predict(X_test)
    error_train[idx] = accuracy_score(y_train, y_train_pred)
    error_val[idx] = accuracy_score(y_test, ypred)
    idx+=1
plot_lambda_selection(reg_vec,error_train,error_val)
plt.savefig('music_classificaiton_ceps.pdf')
print "train accuracy:", error_train
print "val accuracy:", error_val







