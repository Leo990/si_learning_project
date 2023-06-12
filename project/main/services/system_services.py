import io
import json
import os
import pandas as pd
from datetime import datetime

from sklearn.model_selection import KFold, train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

from project.main.config.config import config
from project.main.enums.dataset_enums import ModelEnum, EvaluatorEnum, ScalerEnum, PreprocessEnum
from project.main.enums.extension_enum import ExtensionEnum

from project.main.dtos.service_dtos import ParamTrainDTO, DataSetDTO, RecordDTO, ParamPreprocessDTO
from project.main.services import dataset_service as ds
from project.main.services import record_service as rs
from project.main.services import dropbox_service as ddr

model_dict: dict = {
    ModelEnum.NAIVE_BAYES: GaussianNB(),
    ModelEnum.SVM: SVC(kernel=config.get('default', 'svm_kernel')),
    ModelEnum.DECISION_TREE: DecisionTreeClassifier(max_depth=int(config.get('default', 'tree_decision_depth'))),
    ModelEnum.KNN: KNeighborsClassifier(n_neighbors=int(config.get('default', 'neighbors'))),
    ModelEnum.LOGISTIC_REGRESSION: LogisticRegression()
}


def upload_files(file) -> dict:
    dataframe = pd.read_csv(file, delimiter=",")
    dataset_name = file.filename[:4]
    ddr.load_file(file, f"/{dataset_name}/{dataset_name}{ExtensionEnum.CSV}")
    record = RecordDTO(my_data=dataframe.to_json(orient='records'))
    ident = rs.insert(record)
    dataset = DataSetDTO(f"{dataset_name}", record_id=ident)
    return ds.insert(dataset)


def manage_dataset(param_train: ParamTrainDTO):
    dataset_dto: DataSetDTO = ds.find(param_train.ident)
    record_dto: RecordDTO = rs.find(dataset_dto.record_id)
    if dataset_dto is not None and record_dto is not None and record_dto.is_preprocessed:
        dataframe = pd.DataFrame.from_records(data=json.loads(record_dto.my_data))
        x = dataframe.drop([param_train.y_column], axis=1)  # Axis 1 = column
        y = dataframe[param_train.y_column]

        # Obtiene el modelo y su precisión
        acuraccy, model = _evaluate(x, y, param_train.model_enum, param_train.scaler_enum, param_train.evaluator_enum)

        # Almacena el modelo en mongo y en el dropbox
        _build_dataset(dataset_dto, model, param_train, acuraccy)
        ds.update(dataset_dto)
        return {"EXITO": "El dataset ingresado fue entrenado correctamente!!!"}
    elif dataset_dto is None:
        return {"ERROR": "El dataset no se encuentra en la base de datos."}
    elif record_dto is None:
        return {"ERROR": "El registro no se encuentra en la base de datos."}
    else:
        return {"ERROR": "El dataset ingresado no se encuentra preprocesado para hacer el entrenamiento."}


def _evaluate(x, y, model_enum, scaler_enum, evaluator):
    model = model_dict[model_enum]
    # Crear un objeto KFold para generar los índices de los folds
    if evaluator['enum'] == EvaluatorEnum.HOLD_OUT:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=_evaluate_percent(evaluator['value']))
        return _train(model_dict[model_enum], scaler_enum, x_train, x_test, y_train, y_test)
    elif evaluator['enum'] == EvaluatorEnum.CROSS_VALIDATION:
        kfold = KFold(n_splits=_evaluate_value(evaluator['value']))
        dictionary = {}
        # Realizar la validación cruzada
        for train_indices, val_indices in kfold.split(x):
            # Dividir los datos en conjuntos de entrenamiento y validación
            x_train, x_test = x.iloc[train_indices], x.iloc[val_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[val_indices]
            acuraccy, model = _train(model, scaler_enum, x_train, x_test, y_train, y_test)
            dictionary[acuraccy] = model
        # Elimina el modelo que esté sobreentrenado
        dictionary.pop(1)
        # Retorna el modelo con el acuraccy mas alto
        return max(dictionary.items(), key=lambda key: key[1])
    else:
        raise Exception('No se ingreso un enum valido para la evaluacion del algoritmo')


def _evaluate_percent(value) -> float:
    if type(value).__name__ == 'int' or type(value).__name__ == 'float':
        return float(value) if 1 > float(value) > 0 else 0.2
    raise Exception('Valor ingresado no valido')


def _evaluate_value(value) -> float:
    if type(value).__name__ == 'int' or type(value).__name__ == 'float':
        return int(value) if int(value) >= 1 else 1
    raise Exception('Valor ingresado no valido')


def _train(model, scaler_enum, x_train, x_test, y_train, y_test):
    scaler = MinMaxScaler() if scaler_enum == ScalerEnum.MIN_MAX_SCALER else StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if model is not None:
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        return accuracy_score(y_test, y_predict), model


def _build_dataset(dataset_dto, model, param_train, accuracy):
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


def preprocess_dataset(param_pre_process_dto: ParamPreprocessDTO):
    dataset_dto: DataSetDTO = ds.find(param_pre_process_dto.ident)
    record_dto: RecordDTO = rs.find(dataset_dto.record_id)
    if dataset_dto is not None and record_dto is not None and record_dto.is_preprocessed is False:
        dataframe = pd.DataFrame.from_records(data=json.loads(record_dto.my_data))

        if param_pre_process_dto.preprocess_enum == PreprocessEnum.MEAN:
            preprocess_dataframe = _impute_mean(dataframe)
            pass
        elif param_pre_process_dto.preprocess_enum == PreprocessEnum.MEDIAN:
            preprocess_dataframe = _impute_median(dataframe)
            pass
        elif param_pre_process_dto.preprocess_enum == PreprocessEnum.HOT_DECK:
            preprocess_dataframe = _impute_hot_deck(dataframe)
            pass
        else:
            raise Exception('')
        _build_preprocessed_dataset(preprocess_dataframe, dataset_dto, record_dto)
        return
    else:
        raise Exception('')


def _impute_mean(dataset: pd.DataFrame):
    # Calcular la media de cada columna
    column_means = dataset.mean()

    # Imputar los valores faltantes con la media de cada columna
    return dataset.fillna(column_means)


def _impute_median(dataset: pd.DataFrame):
    # Calcular la mediana de cada columna
    column_means = dataset.median()

    # Imputar los valores faltantes con la media de cada columna
    return dataset.fillna(column_means)


def _impute_hot_deck(dataset):
    df_imputed = dataset.copy()
    # Iterar sobre las columnas
    for col in df_imputed.columns:
        # Obtener los índices de los valores faltantes en la columna
        missing_indices = df_imputed[col].isnull()
        # Obtener los valores no nulos en la misma columna
        non_missing_values = df_imputed.loc[~missing_indices, col]
        # Imputar los valores faltantes con valores no nulos al azar
        df_imputed.loc[missing_indices, col] = non_missing_values.sample(n=missing_indices.sum(), replace=True).values
    return df_imputed


def _build_preprocessed_dataset(preprocess_dataframe, dataset_dto, record_dto):
    record_dto.my_data = preprocess_dataframe.to_json(orient='records')
    record_dto.is_preprocessed = True
    rs.update(record_dto)
    csv_data = preprocess_dataframe.to_csv(index=False)

    # Cargar el archivo a Dropbox
    with io.BytesIO(csv_data.encode()) as stream:
        ddr.load_file(stream.read(), f"/{dataset_dto.name}/{dataset_dto.name}_preprocess{ExtensionEnum.CSV}")
