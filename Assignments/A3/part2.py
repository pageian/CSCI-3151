'''
    Naive Bayes Classifier for spam filtering
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":

    # get file data
    data_file = pd.read_csv("data-sets/emails.csv")
    shuffled_data = data_file.sample(frac=1)
    X = shuffled_data.iloc[1:,0] # Features
    y = shuffled_data.iloc[1:,1] # Target variable

    # vectorize and split data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names()
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=0)
    tot_train = np.append(X_train, y_train[:,None], axis=1)

    # train model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
   
    print("P(Not Spam): " + str(gnb.class_prior_[0]))
    print("P(Spam): " + str(gnb.class_prior_[1]) + "\n")

    # separating spam and non-spam instances
    not_spam = tot_train[np.where(tot_train[:, -1] == 0), :-1][0]
    spam = tot_train[np.where(tot_train[:, -1] == 1), :-1][0]

    # smoothing probs
    not_spam = not_spam + 1
    spam = spam + 1
    X_train_smooth = X_train + 1

    # calculating conditional prob elems
    feature_prob = (X_train_smooth.sum(axis=0)) / len(X_train[:,1])
    feature_prob_spam = (spam.sum(axis=0) / len(spam[:,1]))
    feature_prob_not_spam = (not_spam.sum(axis=0) / len(not_spam[:,1]))

    # printing results
    print("P(Word | Not Spam): " + str(np.prod(feature_prob_not_spam)))
    print("P(Word | Spam): " + str(np.prod(feature_prob_spam)) + "\n")
    print("P(Not Spam | Word): " + str((np.prod(feature_prob_not_spam) * gnb.class_prior_[0]) / np.prod(feature_prob)))
    print("P(Spam | Word): " + str((np.prod(feature_prob_spam) * gnb.class_prior_[1]) / np.prod(feature_prob)))