
from langchain.document_loaders import (TextLoader,
                                        CSVLoader,
                                        PyPDFLoader,
                                        Docx2txtLoader,
                                        UnstructuredPowerPointLoader,
                                        UnstructuredExcelLoader,
                                        UnstructuredEmailLoader
)

def get_max_token_for_model(model, ratio=0.7, prompt_length=2048):
    """Returns the max token for a given model
    Args:
        model (str): model
        ratio (float, optional): ratio of the max token to return. Defaults to 0.7.
        prompt_length (int, optional): length of the prompt. Defaults to 2048.
    Raises:
        Exception: if the model is not supported
    Returns:
        int: max token
    """
    if model is None:
        return 512

    if model == "gpt-3.5-turbo":
        return int(ratio * 4096)-prompt_length
    elif model == "gpt-3.5-turbo-16k":
        return int(ratio * 16384)-prompt_length
    elif model == "gpt-4":
        return int(ratio * 8192)-prompt_length
    elif model == "gpt-4-32k":
        return int(ratio * 32768)-prompt_length
    else:
        raise Exception("Model not supported")


def get_file_type(file):
    """Get the file type from the uploaded file.
    Args:
        file (FileUploader): file uploaded
    Raises:
        Exception: if the file type is not supported
    Returns:
        str: file type
    """

    if file.type == "application/pdf":
        return "pdf"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    elif file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        return "pptx"
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return "xlsx"
    elif file.type == "text/plain":
        return "txt"
    elif file.type == "text/csv":
        return "csv"
    #support email files
    elif file.type == "message/rfc822":
        return "eml"
    else:
        raise Exception("File type not supported")
    
def loader_from_file_from_extension(file, file_type):
    """Returns the langcahin loader for a given file type.
    Args:
        file (FileUploader): file uploaded
        file_type (str): file type
    Raises:
        Exception: if the file type is not supported
    Returns:
        LangchainLoader: langchain loader
    """

    if file_type == "pdf":
        return PyPDFLoader(file)
    elif file_type == "docx":
        return Docx2txtLoader(file)
    elif file_type == "pptx":
        return UnstructuredPowerPointLoader(file)
    elif file_type == "xlsx":
        return UnstructuredExcelLoader(file)
    elif file_type == "txt":
        return TextLoader(file)
    elif file_type == "csv":
        return CSVLoader(file)
    elif file_type == "eml":
        return UnstructuredEmailLoader(file)
    else:
        raise Exception("File type not supported")

