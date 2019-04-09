from Src import Data_Preprocessing
from Src import plot_helpers
from Src.Classifiers import Naive_Bayes_Classifier, SVM_Classifier, Passive_Agressive_Classifier
from Src.Time_checker import run_time_stats
from sklearn import metrics
import numpy as np
import warnings

warnings.filterwarnings("ignore")

actual_test_labels = None
accuracy_dict_naive_bayes = None
accuracy_dict_SVM = None
accuracy_dict_PA = None

Naive_obj = None
SVM_obj = None
PA_obj = None

def Naive_Bayes_Checker():
    '''
    This method runs the Naive bayes models based on count, tfidf and hash vectorizing.
    :return: return the predictions(numpy array with predicted values) of the best of all (count, tfidf, hash).
    '''
    global actual_test_labels, accuracy_dict_naive_bayes, Naive_obj
    predictions = {'count' : tuple() , 'tfidf' : tuple(), 'hash' : tuple()}
    n = Naive_obj = Naive_Bayes_Classifier()

    predictions['count'] = n.Count_vectorizer_classifier()
    predictions['tfidf'] = n.Tfif_vectorizer_classifier()
    predictions['hash'] = n.Hash_vectorizer_classifier()

    actual_test_labels = predictions['count'][0]

    accuracy_dict_naive_bayes = analysis('NAIVE BAYES', predictions)
    key = [key for key,value in accuracy_dict_naive_bayes.items() if value == max(accuracy_dict_naive_bayes.values())][0]

    #returning the predictions of the best predicted model
    return predictions[key][1]

def SVM_Checker():
    '''
    This method runs the Naive bayes models based on count, tfidf and hash vectorizing.
    :return: return the predictions(numpy array with predicted values) of the best of all (count, tfidf, hash).
    '''
    global accuracy_dict_SVM, SVM_obj
    predictions = {'count' : tuple() , 'tfidf' : tuple(), 'hash' : tuple()}
    n = SVM_obj =SVM_Classifier()

    predictions['count'] = n.Count_vectorizer_classifier()
    predictions['tfidf'] = n.Tfif_vectorizer_classifier()
    predictions['hash'] = n.Hash_vectorizer_classifier()

    accuracy_dict_SVM = analysis('SVM', predictions)
    key = [key for key,value in accuracy_dict_SVM.items() if value == max(accuracy_dict_SVM.values())][0]

    #returning the predictions of the best predicted model
    return predictions[key][1]

def Passive_Agressive_Checker():
    '''
        This method runs the Naive bayes models based on count, tfidf and hash vectorizing.
        :return: return the predictions(numpy array with predicted values) of the best of all (count, tfidf, hash).
        '''
    global accuracy_dict_PA, PA_obj
    predictions = {'count': tuple(), 'tfidf': tuple(), 'hash': tuple()}
    n = PA_obj =Passive_Agressive_Classifier()

    predictions['count'] = n.Count_vectorizer_classifier()
    predictions['tfidf'] = n.Tfif_vectorizer_classifier()
    predictions['hash'] = n.Hash_vectorizer_classifier()

    accuracy_dict_PA = analysis('PASSIVE AGRESSIVE', predictions)
    key = [key for key, value in accuracy_dict_PA.items() if value == max(accuracy_dict_PA.values())][0]

    # returning the predictions of the best predicted model
    return predictions[key][1]

def analysis(model,predictions):
    print("######################## {0} ANALYSIS ########################\n".format(model))

    #Calculating the accuracy of the 3 models.
    accuracy_for_count = metrics.accuracy_score(predictions['count'][0] , predictions['count'][1])
    accuracy_for_tfidf = metrics.accuracy_score(predictions['tfidf'][0] , predictions['tfidf'][1])
    accuracy_for_hash = metrics.accuracy_score(predictions['hash'][0] , predictions['hash'][1])

    print("Model accuracy with Count Vectorizer : ", accuracy_for_count*100)
    print("Model accuracy with TFIDF Vectorizer : ", accuracy_for_tfidf*100)
    print("Model accuracy with Hash Vectorizer : ", accuracy_for_hash*100)

    print("\n######################################################################")
    return {'count' : accuracy_for_count , 'tfidf' : accuracy_for_tfidf, 'hash' : accuracy_for_hash}

def voting_classifier(Naive_bayes_predicts, SVM_predicts, Passive_predicts):
    d = {'FAKE' : 0, 'REAL' : 1}
    final_predictions = ['REAL' if d[x]+d[y]+d[z] > 1 else 'FAKE' for x,y,z in zip(Naive_bayes_predicts,SVM_predicts,Passive_predicts)]
    return np.array(final_predictions)

if __name__ == "__main__":
    Naive_bayes_max_predict_result = Naive_Bayes_Checker()
    SVM_max_predict_result  = SVM_Checker()
    Passive_max_predict_result = Passive_Agressive_Checker()
    final_predicts = voting_classifier(Naive_bayes_max_predict_result, SVM_max_predict_result, Passive_max_predict_result)

    print("Final Accuracy is : ", metrics.accuracy_score(actual_test_labels , final_predicts))

    print("---------------------------------------------------------------------------------------")
    run_time_stats(Naive_obj, SVM_obj, PA_obj)

    #Plotting bar chart for the accuracy
    plot_helpers.generate_plot(accuracy_dict_naive_bayes, accuracy_dict_SVM, accuracy_dict_PA)

    #Plotting confusion matrix for final_predictions
    plot_helpers.plot_confusion_matrix(metrics.confusion_matrix( actual_test_labels, final_predicts))