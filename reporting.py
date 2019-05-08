import config
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn import metrics
from sklearn import tree,svm
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from sklearn.externals import joblib

def plot_confusion_matrix(cm, classes=['Malicious','Not Malicious'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


dataset = pd.read_csv("datasets/phishcoop.csv")
dataset = dataset.drop('id', 1) #removing unwanted column

x = dataset.iloc[ : , :-1].values
y = dataset.iloc[:, -1:].values
x = x[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 27, 29]]
print(x.shape)


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0) #change test_size to increase/decrease test sample size

classifier = RandomForestClassifier(n_estimators = 100, criterion = "gini", max_features = 'log2',  random_state = 0)
classifier.fit(X_train, y_train)
joblib.dump(classifier, 'rf_final.pkl')

#predicting the tests set result
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
plot_confusion_matrix(cm,title='Random Forest Confusion matrix')
plt.show()


clf_dtree = tree.DecisionTreeClassifier(max_depth=45,random_state=True)
clf_dtree.fit(X_train,y_train)
prediction = clf_dtree.predict(X_test)
print("Accuracy of decision tree is:",accuracy_score(y_test, prediction))
print("Classification report for decision tree:\n",classification_report(y_test, prediction))
cnf_matrix = metrics.confusion_matrix(y_test,prediction)
plot_confusion_matrix(cnf_matrix,title='DecisionTree Confusion matrix, without normalization')
plt.show()



clf = svm.LinearSVC(C=0.5,random_state=True,multi_class='crammer_singer')
# clf = svm.SVC(gamma=2,C=100,random_state=True)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print("SVM accuracy: ",accuracy_score(y_test, prediction))
cnf_matrix = metrics.confusion_matrix(y_test,prediction)
print("Classification report for svm:\n",classification_report(y_test, prediction))
print(cnf_matrix)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, title='SVM Confusion matrix, without normalization')
plt.show()


clf = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=10000)
clf.fit(X_train,y_train)

prediction = clf.predict(X_test)
print("Neural Network accuracy: ",accuracy_score(y_test, prediction))
print("Classification report for Neural Network:\n",classification_report(y_test, prediction))
cnf_matrix = metrics.confusion_matrix(y_test,prediction)
plot_confusion_matrix(cnf_matrix,title='NeuraLNet Confusion matrix, without normalization')
plt.show()




