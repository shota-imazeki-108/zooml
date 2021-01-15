from main.tree.random_forest_classifier_model import RandomForestClassifierModel
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TestRandomForestClassifierModel:
    iris_dataset = load_iris()
    feature_df = pd.DataFrame(iris_dataset['data'])
    target = iris_dataset['target']
    train_X, valid_X, train_y, valid_y = train_test_split(feature_df, target, test_size=0.2)
    model = RandomForestClassifierModel()
    model.set_params(random_state=0)
    model.fit(train_X, train_y)

    def test_predict(self):
        output = self.model.predict(self.valid_X)
        assert len(self.valid_y) == len(output)

    def test_predict_proba(self):
        output = self.model.predict_proba(self.valid_X)
        assert output.shape == (30, 3)

    def test_predict_score(self):
        output = self.model.predict(self.valid_X)
        assert accuracy_score(self.valid_y, output) > 0.9  # BaseLine Score
