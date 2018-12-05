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
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
# from data import Dataset
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

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
        self.clf_report = []
        

    def compare_clf(self):
        score_df = pd.DataFrame({'Classifiers': self.clf_names, 'Score': self.clf_scores}).set_index('Classifiers')
        print score_df
        for i,c in enumerate(self.clf_report):
            print "Classifier:"+self.clf_names[i]
            print c

    def train_clf(self, clf):
        clf.fit(self.train_X, self.train_y)
        predictions = clf.predict(self.test_X)
        score = accuracy_score(predictions, self.test_y)
        cl_report = classification_report(self.test_y, predictions)
        name = clf.__class__.__name__
        self.clf_scores.append(score)
        self.clf_names.append(name)
        self.clf_report.append(cl_report)

    def svm(self):
        svc_model_linear = SVC(kernel = 'linear', C = 1).fit(self.train_X, self.train_y)
        svc_predictions = svc_model_linear.predict(self.test_X)
        # print precision_recall_fscore_support(self.test_y, svc_predictions, average='macro')
        print classification_report(self.test_y, svc_predictions)

       
class SVM_Hand_Crafted(object):
    def __init__(self, train_path=None, test_path=None):
       
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self._prepare_data()

    def _prepare_data(self):
        self.train_X = self.train.drop('Activity',1)
        self.train_y = self.train.Activity
        self.test_X = self.test.drop('Activity',1)
        self.test_y = self.test.Activity
        
    def svm(self):
        svc=SVC(kernel='linear') 
        svc.fit(self.train_X,self.train_y)
        svc_predictions=svc.predict(self.test_X)
        print('Classification report SVM on Hand Crafted Dataset:')
        print classification_report(self.test_y, svc_predictions)
    
    def pca(self):
        ds = self.train_X.copy()
        ds = ds.append(self.test_X)
        kernel_pca = KernelPCA(kernel="rbf",gamma = 10, fit_inverse_transform = True)
        kernel_pca.fit_transform(ds)
        train_X = ds[:self.train_X.shape[0]]
        test_X = ds[self.train_X.shape[0]:]
        self.train_X = train_X
        self.test_X = test_X

    def rfecv(self):
        rfecv = RFECV(estimator=SVC(kernel = "linear"), step=1, cv=StratifiedKFold(10),
              scoring='accuracy')
        rfecv.fit(self.train_X, self.train_y)
        print "Best number of features:" + str(rfecv.n_features_)
        print "Accuracy on test data:" + str(rfecv.score(self.test_X,self.test_y))
        print "RFECV feature ranking:" 
        print rfecv.ranking_
        


        
        
        

if __name__ == "__main__":
    dataset_root = "/media/saagar14/DATA/Cloud_Computing/Semester 3/ML/mlproj/HAR/code/dataset/UCI HAR Dataset"
    dataset = Dataset(dataset_root=dataset_root)

    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    bm = Baseline(train_X,train_y,test_X,test_y)
    for clf in bm.classifiers:
        bm.train_clf(clf)


    # Compare baseline classifiers
    bm.compare_clf()
    
    # SVM on preprocessed data (Without Handcrafted features)
    bm.svm()

    # SVM on hand crafted features

    # fe_dataset_root = "dataset/Hand-Crafted"
    # svm_fe = SVM_Hand_Crafted(fe_dataset_root+"/train.csv",fe_dataset_root+"/test.csv")
    
    #PCA
    
    # svm_fe.pca()
    # svm_fe.svm()

    #RFECV
    
    # svm_fe.rfecv()