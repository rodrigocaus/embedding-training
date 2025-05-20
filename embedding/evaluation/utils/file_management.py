import os


def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def format_evaluation_name(model_path, original):
    base_path = model_path.split("/")[1]
    if original:
        base_path = "original_" + base_path
    return base_path
