import json
import os

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
from project.main.enums.extension_enum import ExtensionEnum
from project.main.dtos.param_train_dto import ParamTrainDTO
from project.main.dtos.dataset_dto import DataSetDTO
from project.main.services import dataset_service as ds
from project.main.config.config import config
from project.main.dtos.record_dto import RecordDTO
from project.main.services import record_service as rs
from project.main.services import dropbox_service as ddr

model_dict: dict = {
    ModelEnum.NAIVE_BAYES: GaussianNB(),
    ModelEnum.SVM: SVC(kernel=config.get('default', 'svm_kernel')),
    # ModelEnum.DECISION_TREE: DecisionTreeClassifier(max_depth=int(config.get('default', 'tree_decision_depth')))
}


def upload_files(file) -> dict:
    dataframe = pd.read_csv(file, delimiter=",")
    dataset_name = file.filename[:4]
    ddr.load_file(file, f"/{dataset_name}/{dataset_name}{ExtensionEnum.CSV}")
    record = RecordDTO(my_data=dataframe.to_json(orient='records'))
    ident = rs.insert(record)
    dataset = DataSetDTO(f"{dataset_name}", record_id=ident)
    return ds.insert(dataset)


def train(param_train: ParamTrainDTO):
    dataset_dto: DataSetDTO = ds.find(param_train.ident)
    record_dto: RecordDTO = rs.find(dataset_dto.record_id)
    if dataset_dto is not None and record_dto is not None and record_dto.is_preprocessed:
        dataframe = pd.DataFrame.from_records(data=json.loads(record_dto.my_data))
        x = dataframe.drop([param_train.y_column], axis=1)  # Axis 1 = column
        y = dataframe[param_train.y_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=param_train.test_size)
        scaler = MinMaxScaler() if param_train.scaler_enum == ScalerEnum.MIN_MAX_SCALER else StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = model_dict[param_train.model_enum]
        if model is not None:
            model.fit(x_train, y_train)
            y_predict = model.predict(x_test)
            build_dataset(dataset_dto, model, param_train, accuracy_score(y_test, y_predict))
            ds.update(dataset_dto.ident, dataset_dto)
            return {"EXITO": "El dataset ingresado fue entrenado correctamente!!!"}
        else:
            pass
    elif dataset_dto is None:
        return {"ERROR": "El dataset no se encuentra en la base de datos."}
    elif record_dto is None:
        return {"ERROR": "El registro no se encuentra en la base de datos."}
    else:
        return {"ERROR": "El dataset ingresado no se encuentra preprocesado para hacer el entrenamiento."}


def build_dataset(dataset_dto, model, param_train, accuracy):
    temp_path = f"./{param_train.model_enum}"
    final_path = f"/{dataset_dto.name}/{param_train.model_enum}/{datetime.now().date().year}{ExtensionEnum.H5}"
    model.save(temp_path)
    # Sube el archivo del modelo a Dropbox
    with open(temp_path, 'rb') as f:
        ddr.load_file(f.read(), final_path)
    if os.path.isfile(temp_path):
        os.remove(temp_path)
    dataset_dto.model_name = param_train.model_enum
    dataset_dto.accuracy = accuracy
