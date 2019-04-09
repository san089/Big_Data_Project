import matplotlib.pyplot as plt
import numpy as np

def run_time_stats(Naive_obj, SVM_obj, PA_obj):
    '''
    This method take the input as the objects of all the 3 classifiers and prints the time statistics
    :param Naive_obj: Object of Naive Bayes Classifier
    :param SVM_obj: Object of SVM Classifier
    :param PA_obj: Object of Passive Agressive Classifier
    :return: None
    '''
    vectorizer_time = Naive_obj.vectorizer_time
    Naive_timer = Naive_obj.classifier_time
    SVM_timer = SVM_obj.classifier_time
    PA_timer = PA_obj.classifier_time

    print("\n\n######################## Vectorizer Time Stats ########################\n")
    print("Time Taken by Vectorizers\n")
    print("Count Vectorizer : {0}\nTFIDF Vectorizer : {1}\nHash Vectorizer : {2}".format(vectorizer_time['Count'], vectorizer_time['TFIDF'],vectorizer_time['Hash']  ))

    print("\n\n######################## Classifier Time Stats ########################\n")
    print("\nNAIVE BAYES\nTime taken with Count Vectorizer : {0}".format(Naive_timer['Count']))
    print("Time taken with TFIDF Vectorizer : {0}".format(Naive_timer['TFIDF']))
    print("Time taken with Hash Vectorizer : {0}".format(Naive_timer['Hash']))

    print("\nSVM\nTime taken with Count Vectorizer : {0}".format(SVM_timer['Count']))
    print("Time taken with TFIDF Vectorizer : {0}".format(SVM_timer['TFIDF']))
    print("Time taken with Hash Vectorizer : {0}".format(SVM_timer['Hash']))

    print("\nPassive Agressive\nTime taken with Count Vectorizer : {0}".format(PA_timer['Count']))
    print("Time taken with TFIDF Vectorizer : {0}".format(PA_timer['TFIDF']))
    print("Time taken with Hash Vectorizer : {0}".format(PA_timer['Hash']))

    plot_vectorizer_time(vectorizer_time)
    plot_classifier_times(Naive_timer, SVM_timer, PA_timer)
    return None


def plot_vectorizer_time(vectorizer_time):
    f, ax = plt.subplots()

    ax.set_xticks(range(1,3))
    plt.title("Vectorizer Execution Time")
    plt.xlabel("Vectorizer Type")
    plt.ylabel("Time (in secs)")

    plt.bar(range(len(vectorizer_time)), list(vectorizer_time.values()), align='center', color = '#7663b0', width=0.5 , edgecolor = 'black')
    plt.xticks(range(len(vectorizer_time)), list(vectorizer_time.keys()))

    for index,val in enumerate(vectorizer_time.values()):
        ax.text(index , val + 1  , horizontalalignment = 'center' , s = round(val , 3), color = 'red' , fontweight = 'bold')

    ax.set_ylim([0,80])
    f.set_dpi(75)
    f.set_frameon(True)
    plt.show()


def plot_classifier_times(Naive_timer, SVM_timer, PA_timer):
    count = [list(Naive_timer.values())[0] , list(SVM_timer.values())[0], list(PA_timer.values())[0]]
    tfidf = [list(Naive_timer.values())[1] , list(SVM_timer.values())[1], list(PA_timer.values())[1]]
    hash = [list(Naive_timer.values())[2] , list(SVM_timer.values())[2], list(PA_timer.values())[2]]

    #Rounding off the time values
    count = [round(val,3) for val in count]
    tfidf = [round(val,3) for val in tfidf]
    hash = [round(val,3) for val in hash]


    fig, ax = plt.subplots()
    ind = np.arange(3)
    bar_width = 0.2

    p1 = plt.bar(ind , count, bar_width, color = 'r' )
    p2 = plt.bar(ind + bar_width, tfidf, bar_width, color = 'b' )
    p3 = plt.bar(ind + + bar_width + bar_width, hash, bar_width, color = 'g')

    for plot in [p1,p2,p3]:
        for rect in plot:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%s' % round(float(height),2), ha='center', va='bottom')

    ax.set_title('Execution Time of Classifiers')
    ax.set_xticks(ind + bar_width)
    ax.set_xticklabels(('Naive Bayes', 'SVM', 'Passive Agressive'))
    plt.legend(['Count', 'TFIDF', 'Hash'])
    ax.set_axisbelow(True)
    plt.show()

'''
N = {'Count' : 0.05902385711669922 , 'TFIDF' :  0.40993690490722656, 'Hash' : 0.087799072265625}
S = {'Count' :  7.1091132164001465, 'TFIDF' :  5.953486919403076, 'Hash' : 2.0117762088775635}
P = {'Count' :  0.24056363105773926, 'TFIDF' :  3.281041383743286, 'Hash' : 0.3645296096801758}

plot_classifier_times(N, S, P)

'''