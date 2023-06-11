class DataSetDTO:
    def __init__(self, name, records, model_name=None, accuracy=0.0, is_preprocessed=False, ident=None):
        self.id = ident
        self.name = name
        self.records = records
        self.model_name = model_name
        self.accuracy = accuracy
        self.is_preprocessed = is_preprocessed

    def serialize(self, have_id: bool):
        return {
            "id": self.id,
            "name": self.name,
            "records": self.records,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "is_preprocessed": self.is_preprocessed
        } if have_id else {
            "name": self.name,
            "records": self.records,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "is_preprocessed": self.is_preprocessed
        }
