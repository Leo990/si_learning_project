import pandas as pd
from datetime import datetime

from project.main.dtos.dataset_dto import DataSetDTO
from project.main.services import dataset_service as ds


def upload_file(file) -> dict:
    dataframe = pd.read_csv(file, delimiter=",")
    dataset = DataSetDTO(f"{file.filename[:4]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                         dataframe.to_json(orient='records'))
    return ds.insert(dataset)
