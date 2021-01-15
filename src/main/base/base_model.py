from sklearn.base import BaseEstimator
from abc import abstractmethod


class BaseModel(BaseEstimator):
    def __init__(self):
        self.__params = {}

    def get_params(self):
        return self.__params

    def set_params(self, **params):
        for k, v in params.items():
            self.__params[k] = v

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
