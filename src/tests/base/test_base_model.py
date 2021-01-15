from main.base.base_model import BaseModel


class TestBaseModel:
    def test_get_params(self):
        model = BaseModel()
        assert model.get_params() == {}

    def test_set_params(self):
        model = BaseModel()
        model.set_params(test='test')
        assert model.get_params() == {'test': 'test'}
