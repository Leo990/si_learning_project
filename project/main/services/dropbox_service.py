import dropbox.files as fl
from project.main.config.config import DropBoxContext
from dropbox.exceptions import ApiError

DROPBOX_CONTEXT = DropBoxContext()
dbx = DROPBOX_CONTEXT.dbx


def load_file(file, path: str):
    try:
        # Cargar el archivo en Dropbox
        dbx.files_upload(file.read(), path, mode=fl.WriteMode.overwrite)
        return 'Archivo subido con éxito a Dropbox'
    except Exception as e:
        # Capturar cualquier otra excepción no manejada anteriormente
        print(e)
        return f'Ocurrió un error inesperado: {str(e)}'


def download_file(path: str):
    try:
        # Descarga el archivo de Dropbox
        metadata, response = dbx.files_download(path)
        # Obtén el contenido del archivo y su nombre

        return metadata,response

    except ApiError as e:
        return f'Error al descargar el archivo: {str(e.error)}'


""""Lista los archivos de dropbox"""
def list_files(path:str):
    try:
        files = []
        # Get the list of files in Dropbox
        result = dbx.files_list_folder(path)
        for entry in result.entries:
            files.append(entry.name)

        return files

    except ApiError as e:
        return f'Error al listar los archivos {str(e.error)}'

