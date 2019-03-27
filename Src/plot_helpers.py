import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sn

def generate_plot(nv, svm, pa):
    count = [nv['count'], svm['count'], pa['count']]
    tfidf = [nv['tfidf'], svm['tfidf'], pa['tfidf']]
    hashed = [nv['hash'], svm['hash'], pa['hash']]

    fig, ax = plt.subplots(figsize=(10, 6))
    num_class = 3
    x = np.arange(num_class)
    wid = 0.2

    p1 = ax.bar(x, count, width=wid, color='orangered', bottom=0)
    p2 = ax.bar(x + wid, tfidf, width=wid, color='slateblue', bottom=0)
    p3 = ax.bar(x + 2 * wid, hashed, width=wid, color='chartreuse', bottom=0)

    ax.set_title('Plot')

    # Plot labels
    plt.title("Performance Graph")
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy Score")
    plt.xticks(x + wid, ['Naive Bayes', 'SVM', 'Passive Agressive'])

    # Plot legends
    red_patch = mpatches.Patch(color='orangered', label='Count Vectorizer')
    blue_patch = mpatches.Patch(color='slateblue', label='TFIDF Vectorizer')
    green_patch = mpatches.Patch(color='chartreuse', label='Hash Vectorizer')
    plt.legend(bbox_to_anchor=(1, 1), handles=[red_patch, blue_patch, green_patch])

    plt.show()

def plot_confusion_matrix(matrix):
    labels = ['REAL', 'FAKE']

    ax = plt.subplot()

    df_cm = pd.DataFrame(matrix, index=["Predicted REAL", "Predicted FAKE"],
                         columns=["Actual REAL", "Actual FAKE"])

    sn.heatmap(df_cm, ax=ax, annot=True, fmt='d', center=0.1, linewidths=2, cmap="RdBu_r", linecolor='black')

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels')
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()