from sklearn.base import ClassifierMixin
from abc import abstractmethod
from main.base.base_model import BaseModel


class BaseClassifier(BaseModel, ClassifierMixin):
    @abstractmethod
    def predict_proba(self, X):
        pass
