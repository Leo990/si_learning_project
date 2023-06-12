class RecordDTO:
    def __init__(self, my_data, is_preprocessed: bool = False, ident: str = None):
        self.my_data = my_data
        self.is_preprocessed = is_preprocessed
        self.ident = ident

    def serialize(self, have_id: bool):
        return {
            "ident": self.ident,
            "my_data": self.my_data,
            "is_preprocessed": self.is_preprocessed
        } if have_id else {
            "my_data": self.my_data,
            "is_preprocessed": self.is_preprocessed
        }


class ParamTrainDTO:

    def __int__(self, ident, y_column, test_size, scaler_enum, model_enum, evaluator_enum):
        self.ident = ident
        self.y_column = y_column
        self.test_size = test_size
        self.scaler_enum = scaler_enum
        self.model_enum = model_enum
        self.evaluator_enum = None


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
