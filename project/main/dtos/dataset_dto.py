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
