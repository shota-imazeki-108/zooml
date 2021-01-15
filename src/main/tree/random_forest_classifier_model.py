from main.base.base_classifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifierModel(BaseClassifier):

    def fit(self, X, y):
        if len(self.get_params().keys()) > 0:
            self.clf = RandomForestClassifier(**self.get_params())
        else:
            self.clf = RandomForestClassifier()
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
