# Big_Data_Project

Fake News Detection Project.

In this project we displayed how to use machine learning algorithms for text classification. 

We worked on classifying whether the given news article is fake or real. 

Data Cleaning and preprocessing: 

Removed special characters from text
Spell checked all the documents
Removed the Stop Words
Vectorized the documents.

Vectorization
For Vectorization we used - Count Vectorizer, TFIDF Vectorizer, Hash Vectorizer.

For Classification prurpose we Used: Multinomial Naive Bayes, Support Vector Machine ( LinearSVC ), PassiveAgressiveClassifier.

We compared the performance of the vectorizers as well as the classifiers. 

At the end, we are using an ensemble model to get a higher accuracy. We use scikit-learn max voting classifier for it.
