from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from Src import Data_Preprocessing
import warnings
import time

warnings.filterwarnings("ignore")

data_obj = Data_Preprocessing.process_data()
data_obj.clean_data()

'''
Naive Bayes classifier class
'''
class Naive_Bayes_Classifier:

    def __init__(self):
        self.vectorizer_time = {'Count' : 0 , 'TFIDF' : 0 , 'Hash' : 0}
        self.classifier_time = {'Count' : 0 , 'TFIDF' : 0 , 'Hash' : 0}

    def Count_vectorizer_classifier(self):

        start = time.time()

        print("\n\nRunning Naive Bayes with Count Vectorizer...")
        x_train, x_test = data_obj.generate_count_vectorizer()
        y_train = data_obj.y_train
        y_test = data_obj.y_test

        self.vectorizer_time['Count'] = time.time() - start

        start = time.time()
        pred = self.classify(x_train, x_test, y_train, y_test)
        self.classifier_time['Count'] = time.time() - start

        print("Process Completed.")

        return (y_test, pred)

    def Tfif_vectorizer_classifier(self):
        start = time.time()

        print("\n\nRunning Naive Bayes with TFIDF Vectorizer...")
        x_train, x_test = data_obj.generate_tfidf_vectorizer()
        y_train = data_obj.y_train
        y_test = data_obj.y_test

        self.vectorizer_time['TFIDF'] = time.time() - start

        start = time.time()
        pred = self.classify(x_train, x_test, y_train, y_test)
        self.classifier_time['TFIDF'] = time.time() - start

        print("Process Completed.")
        return (y_test, pred)

    def Hash_vectorizer_classifier(self):
        start = time.time()

        print("\n\nRunning Naive Bayes with Hash Vectorizer...")
        x_train, x_test = data_obj.generate_hashing_vectorizer()
        y_train = data_obj.y_train
        y_test = data_obj.y_test

        self.vectorizer_time['Hash'] = time.time() - start
        start = time.time()
        pred = self.classify(x_train, x_test, y_train, y_test)

        self.classifier_time['Hash'] = time.time() - start
        print("Process Completed.")
        return (y_test, pred)

    def classify(self, x_train, x_test, y_train, y_test):
        model = MultinomialNB()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        return pred
    


'''
SVM classifier class
'''
class SVM_Classifier:

    def __init__(self):
        self.vectorizer_time = {'Count': 0, 'TFIDF': 0, 'Hash': 0}
        self.classifier_time = {'Count': 0, 'TFIDF': 0, 'Hash': 0}

    def Count_vectorizer_classifier(self):
        start = time.time()

        print("\n\nRunning SVM with Count Vectorizer...")
        x_train, x_test = data_obj.generate_count_vectorizer()
        y_train = data_obj.y_train
        y_test = data_obj.y_test

        self.vectorizer_time['Count'] = time.time() - start

        start = time.time()
        pred = self.classify(x_train, x_test, y_train, y_test)
        self.classifier_time['Count'] = time.time() - start

        print("Process Completed.")

        return (y_test, pred)

    def Tfif_vectorizer_classifier(self):
        start = time.time()

        print("\n\nRunning SVM with TFIDF Vectorizer...")
        x_train, x_test = data_obj.generate_tfidf_vectorizer()
        y_train = data_obj.y_train
        y_test = data_obj.y_test

        self.vectorizer_time['TFIDF'] = time.time() - start

        start = time.time()
        pred = self.classify(x_train, x_test, y_train, y_test)
        self.classifier_time['TFIDF'] = time.time() - start

        print("Process Completed.")
        return (y_test, pred)

    def Hash_vectorizer_classifier(self):
        start = time.time()

        print("\n\nRunning SVM with Hash Vectorizer...")
        x_train, x_test = data_obj.generate_hashing_vectorizer()
        y_train = data_obj.y_train
        y_test = data_obj.y_test

        self.vectorizer_time['Hash'] = time.time() - start
        start = time.time()
        pred = self.classify(x_train, x_test, y_train, y_test)

        self.classifier_time['Hash'] = time.time() - start
        print("Process Completed.")
        return (y_test, pred)

    def classify(self, x_train, x_test, y_train, y_test):
        model = LinearSVC( C = 0.1 , loss='hinge', penalty='l2', max_iter=1000, dual= True)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        return pred


'''
Passive Agressive classifier class
'''
class Passive_Agressive_Classifier:

    def __init__(self):
        self.vectorizer_time = {'Count': 0, 'TFIDF': 0, 'Hash': 0}
        self.classifier_time = {'Count': 0, 'TFIDF': 0, 'Hash': 0}

    def Count_vectorizer_classifier(self):
        start = time.time()

        print("\n\nRunning Passive Agressive with Count Vectorizer...")
        x_train, x_test = data_obj.generate_count_vectorizer()
        y_train = data_obj.y_train
        y_test = data_obj.y_test

        self.vectorizer_time['Count'] = time.time() - start

        start = time.time()
        pred = self.classify(x_train, x_test, y_train, y_test)
        self.classifier_time['Count'] = time.time() - start

        print("Process Completed.")

        return (y_test, pred)

    def Tfif_vectorizer_classifier(self):
        start = time.time()

        print("\n\nRunning Passive Agressive with TFIDF Vectorizer...")
        x_train, x_test = data_obj.generate_tfidf_vectorizer()
        y_train = data_obj.y_train
        y_test = data_obj.y_test

        self.vectorizer_time['TFIDF'] = time.time() - start

        start = time.time()
        pred = self.classify(x_train, x_test, y_train, y_test)
        self.classifier_time['TFIDF'] = time.time() - start

        print("Process Completed.")
        return (y_test, pred)

    def Hash_vectorizer_classifier(self):
        start = time.time()

        print("\n\nRunning Passive Agressive with Hash Vectorizer...")
        x_train, x_test = data_obj.generate_hashing_vectorizer()
        y_train = data_obj.y_train
        y_test = data_obj.y_test

        self.vectorizer_time['Hash'] = time.time() - start
        start = time.time()
        pred = self.classify(x_train, x_test, y_train, y_test)

        self.classifier_time['Hash'] = time.time() - start
        print("Process Completed.")
        return (y_test, pred)

    def classify(self, x_train, x_test, y_train, y_test):
        model = PassiveAggressiveClassifier(max_iter=50, C = 0.7)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        return pred

