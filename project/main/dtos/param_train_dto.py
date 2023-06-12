class ParamTrainDTO:
    def __int__(self, ident, y_column, test_size, scaler_enum, model_enum):
        self.ident = ident
        self.y_column = y_column
        self.test_size = test_size
        self.scaler_enum = scaler_enum
        self.model_enum = model_enum
