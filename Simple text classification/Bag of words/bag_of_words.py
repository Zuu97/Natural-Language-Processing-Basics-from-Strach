import numpy as np
import pandas as pd
from util import get_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class BagOfWords(object):
    def __init__(self):
        Xtrain, Ytrain, Xtest, Ytest, encoder = get_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.encoder = encoder

    def rf_model(self):
        model = RandomForestClassifier(n_estimators=200)
        model.fit(self.Xtrain, self.Ytrain)
        self.model = model

    def predict_and_evaluate(self):
        Ytrain_pred = self.model.predict(self.Xtrain)
        train_acc = accuracy_score(self.Ytrain, Ytrain_pred)

        Ytest_pred = self.model.predict(self.Xtest)
        test_acc = accuracy_score(self.Ytest, Ytest_pred)

        print("Train accuracy: ",train_acc)
        print("Test accuracy: ",test_acc)

if __name__ == "__main__":
    classifier = BagOfWords()
    classifier.rf_model()
    classifier.predict_and_evaluate()