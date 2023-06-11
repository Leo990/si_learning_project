import pandas as pd
from datetime import datetime

from project.main.dtos.dataset_dto import DataSetDTO
from project.main.dtos.record_dto import RecordDTO
from project.main.services import record_service as rs
from project.main.services import dataset_service as ds


def __init__(file) -> dict:
    dataframe = pd.read_csv(file, delimiter=",")
    record = RecordDTO(my_data=dataframe.to_json(orient='records'))
    ident = rs.insert(record)
    dataset = DataSetDTO(f"{file.filename[:4]}", record_id=ident)
    return ds.insert(dataset)
