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
