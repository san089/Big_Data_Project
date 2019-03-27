from sklearn.svm import LinearSVC
from sklearn import metrics
from Src import Data_Preprocessing

class SVM_Classifier:

    def __init__(self):
        self.p = Data_Preprocessing.process_data()
        self.p.clean_data()

    def Count_vectorizer_classifier(self):
        x_train, x_test = self.p.generate_count_vectorizer()
        y_train = self.p.y_train
        y_test = self.p.y_test
        pred = self.classify(x_train, x_test, y_train, y_test)
        return (y_test, pred)

    def Tfif_vectorizer_classifier(self):
        x_train, x_test = self.p.generate_tfidf_vectorizer()
        y_train = self.p.y_train
        y_test = self.p.y_test
        pred = self.classify(x_train, x_test, y_train, y_test)
        return (y_test, pred)

    def Hash_vectorizer_classifier(self):
        x_train, x_test = self.p.generate_hashing_vectorizer()
        y_train = self.p.y_train
        y_test = self.p.y_test
        pred = self.classify(x_train, x_test, y_train, y_test)
        return (y_test, pred)


    def classify(self, x_train, x_test, y_train, y_test):
        model = LinearSVC(C=1.0 ,penalty='l1', max_iter=3000, dual= False)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        return pred
