class RecordDTO:
    def __init__(self, my_data, is_preprocessed: bool = False, ident: str = None):
        self.my_data = my_data
        self.is_preprocessed = is_preprocessed
        self.ident = ident

    def __getstate__(self):
        return {
            'my_data': self.my_data,
            'is_preprocessed': self.is_preprocessed
        }

    def __setstate__(self, state):
        self.my_data = state['my_data']
        self.is_preprocessed = state['is_preprocessed']


class ParamTrainDTO:

    def __init__(self, ident: str, y_column: str, scaler_enum: str, model_enum: str, evaluator: dict):
        self.ident = ident
        self.y_column = y_column
        self.scaler_enum = scaler_enum
        self.model_enum = model_enum
        self.evaluator = evaluator


class ParamPreprocessDTO:
    def __init__(self, ident: str, preprocess_enum: str):
        self.ident = ident
        self.preprocess_enum = preprocess_enum


class DataSetDTO:
    def __init__(self, name, record_id, model_name=None, accuracy=0.0, ident=None):
        self.ident = ident
        self.name = name
        self.record_id = record_id
        self.model_name = model_name
        self.accuracy = accuracy

    def serialize(self, have_id: bool):
        return {
            "ident": self.ident,
            "name": self.name,
            "record_id": self.record_id,
            "model_name": self.model_name,
            "accuracy": self.accuracy
        } if have_id else {
            "name": self.name,
            "record_id": self.record_id,
            "model_name": self.model_name,
            "accuracy": self.accuracy
        }


class PredictDTO:

    def __init__(self, ident, array):
        self.ident = ident
        self.array = array


