import pandas as pd
import numpy as np
import itertools
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from tkinter import *

root = Tk()
root.geometry('500x500')

l = Label(root, text="Welcome to detector of Fake News")
l.pack()

root.mainloop()


# Read the data
see = pd.read_csv(r"path to your datafile")

# Get shape and head
# print(see.shape)
# print(see.head())

# DataFlair - Get the labels
labels = see.label
labels.head()

print(labels.head())

# Dataset - Split the dataset
x_train, x_test, y_train, y_test = train_test_split(see['text'], labels, test_size=0.2, random_state=7)

# Dataset - Initialize a TfidfVectorizer
tfidf_vectorizer1 = TfidfVectorizer(stop_words='english', max_df=0.7)

# Dataset - Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer1.fit_transform(x_train)
tfidf_test = tfidf_vectorizer1.transform(x_test)

# Dataset - Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)


# Dataset - Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')



# Dataset - Build confusion matrix
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

# Classification Report - Precision, recall, f1-score, support
print(classification_report(y_test,y_pred))

root.mainloop()