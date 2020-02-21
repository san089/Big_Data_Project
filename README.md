# Big_Data_Project - Fake News Detection

In this project we displayed the use machine learning algorithms for text classification. We worked on classifying whether the given news article is fake or real. 

### Data Cleaning and preprocessing: 
Removed special characters from text
Spell checked all the documents
Removed the Stop Words
Vectorized the documents.

### Vectorization
For Vectorization we used - Count Vectorizer, TFIDF Vectorizer, Hash Vectorizer.

### Classification
For Classification prurpose we Used: Multinomial Naive Bayes, Support Vector Machine ( LinearSVC ), PassiveAgressiveClassifier.
We compared the performance of the vectorizers as well as the classifiers. 
At the end, we are using an ensemble model to get a higher accuracy. We use scikit-learn max voting classifier for it.

### How to run 
Clone the project and simply run the Main.py inside the src folder.

`python Main.py`


## Results

```
[nltk_data] Downloading package stopwords to C:\Users\Sanchit
[nltk_data]     Kumar\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Records Count:  6335
Column Count :  4
Columns :  ['id' 'title' 'text' 'label']
Count of FAKE and REAL labels :
        text
label
FAKE   3164
REAL   3171


Starting Data Cleaning Process.....
Running spell check, stemming and stop word removal.....
Data Cleaning Process Completed.


Running Naive Bayes with Count Vectorizer...
Process Completed.


Running Naive Bayes with TFIDF Vectorizer...
Process Completed.


Running Naive Bayes with Hash Vectorizer...
Process Completed.
######################## NAIVE BAYES ANALYSIS ########################

Model accuracy with Count Vectorizer :  89.04830224772836
Model accuracy with TFIDF Vectorizer :  88.47441415590626
Model accuracy with Hash Vectorizer :  81.34863701578192

######################################################################


Running SVM with Count Vectorizer...
Process Completed.


Running SVM with TFIDF Vectorizer...
Process Completed.


Running SVM with Hash Vectorizer...
Process Completed.
######################## SVM ANALYSIS ########################

Model accuracy with Count Vectorizer :  88.6178861788618
Model accuracy with TFIDF Vectorizer :  90.14825442372072
Model accuracy with Hash Vectorizer :  91.96556671449068

######################################################################


Running Passive Agressive with Count Vectorizer...
Process Completed.


Running Passive Agressive with TFIDF Vectorizer...
Process Completed.


Running Passive Agressive with Hash Vectorizer...
Process Completed.
######################## PASSIVE AGRESSIVE ANALYSIS ########################

Model accuracy with Count Vectorizer :  89.38307030129124
Model accuracy with TFIDF Vectorizer :  92.58727881396462
Model accuracy with Hash Vectorizer :  92.01339072214252

######################################################################
Final Accuracy is :  0.9340028694404591
---------------------------------------------------------------------------------------


######################## Vectorizer Time Stats ########################

Time Taken by Vectorizers

Count Vectorizer : 6.0200676918029785
TFIDF Vectorizer : 59.66688680648804
Hash Vectorizer : 2.35701847076416


######################## Classifier Time Stats ########################


NAIVE BAYES
Time taken with Count Vectorizer : 0.05902385711669922
Time taken with TFIDF Vectorizer : 0.40993690490722656
Time taken with Hash Vectorizer : 0.087799072265625

SVM
Time taken with Count Vectorizer : 7.1091132164001465
Time taken with TFIDF Vectorizer : 5.953486919403076
Time taken with Hash Vectorizer : 2.0117762088775635

Passive Agressive
Time taken with Count Vectorizer : 0.24056363105773926
Time taken with TFIDF Vectorizer : 3.281041383743286
Time taken with Hash Vectorizer : 0.3645296096801758
```
