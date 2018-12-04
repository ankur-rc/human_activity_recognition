'''
Created Date: Thursday November 29th 2018
Last Modified: Thursday November 29th 2018 9:23:41 pm
Author: saagar14
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from data import Dataset

class Baseline(object):

    def __init__(self, train_X=None, train_y=None,test_X=None, test_y=None,verbosity=1):
        self.verbose = verbosity

        self.n_timesteps = train_X.shape[1]
        self.n_features = train_X.shape[2]
        self.n_outputs = test_y.shape[1]

        ds = Dataset()
        # Reshaping the datasets
        self.train_X = train_X.reshape(len(train_X),self.n_timesteps*self.n_features)
        self.train_y = train_y
        self.test_X = test_X.reshape(len(test_X),self.n_timesteps*self.n_features)
        self.test_y = test_y

        self.model = None

        self.classifiers = [KNeighborsClassifier(self.n_outputs+1), GaussianNB(), SVC(), DecisionTreeClassifier()]
        self.clf_names = []
        self.clf_scores = []


    def compare_clf(self):
        score_df = pd.DataFrame({'Classifiers': self.clf_names, 'Score': self.clf_scores}).set_index('Classifiers')
        print score_df

    def train_clf(self, clf):
        clf.fit(self.train_X, self.train_y)
        predictions = clf.predict(self.test_X)
        score = accuracy_score(predictions, self.test_y)
        name = clf.__class__.__name__
        self.clf_scores.append(score)
        self.clf_names.append(name)

    def svm(self):
        svc_model_linear = SVC(kernel = 'linear', C = 1).fit(self.train_X, self.train_y)
        svc_predictions = svc_model_linear.predict(self.test_X)
        print precision_recall_fscore_support(self.test_y, svc_predictions, average='macro')



if __name__ == "__main__":
    dataset_root = "/media/saagar14/DATA/Cloud_Computing/Semester 3/ML/mlproj/HAR/code/dataset/UCI HAR Dataset"
    dataset = Dataset(dataset_root=dataset_root)

    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    bm = Baseline(train_X,train_y,test_X,test_y)
    for clf in bm.classifiers:
        bm.train_clf(clf)
    bm.compare_clf()
    bm.svm()

    
