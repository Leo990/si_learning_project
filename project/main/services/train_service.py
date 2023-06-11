import pandas as pd
from datetime import datetime

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from project.main.enums.model_enum import ModelEnum
from project.main.enums.scaler_enum import ScalerEnum
from project.main.dtos.param_train_dto import ParamTrainDTO
from project.main.dtos.dataset_dto import DataSetDTO
from project.main.services import dataset_service as ds

model_dict: dict = {
    ModelEnum.NAIVE_BAYES: GaussianNB(),
    ModelEnum.SVM: SVC(),
    ModelEnum.DECISION_TREE: DecisionTreeClassifier()
}


def train(param_train: ParamTrainDTO):
    dataset_dto: DataSetDTO = ds.find(param_train.ident)
    if dataset_dto.is_preprocessed:
        dataframe = pd.read_json(dataset_dto.records)
        model_name = f"{param_train.model_enum}_{param_train.scaler_enum}_{datetime.now().date()}"
        x = dataframe.drop([param_train.y_column], axis=1)  # Axis 1 = column
        y = dataframe[param_train.y_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=param_train.test_size)
        scaler = MinMaxScaler() if param_train.scaler_enum == ScalerEnum.MIN_MAX_SCALER else StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = model_dict[param_train.model_enum]
        model.fit(x_train, y_train)
        model.save(model_name)
        y_predict = model.predict(x_test)
        dataset_dto.model_name = model_name
        dataset_dto.accuracy = accuracy_score(y_test, y_predict)
        return {"EXITO": "El dataset ingresado fue entrenado correctamente!!!"}
    else:
        return {"ERROR" : "El dataset ingresado no se encuentra preprocesado para hacer el entrenamiento"}
