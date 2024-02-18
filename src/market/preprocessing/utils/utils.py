def load_yaml(path_file):
    import yaml
    try:
        with open(path_file, 'r') as file:
            configuration = yaml.safe_load(file)
        return configuration
    except FileNotFoundError:
        print('file not found')


def check_mkdir(path_to_save, extra_path=None):
    import os
    if isinstance(path_to_save, str) is False:
        raise ValueError('"path_to_save" must be string')
    try:
        if extra_path is not None:
            os.makedirs(path_to_save + extra_path)
        else:
            os.makedirs(path_to_save)
    except FileExistsError:
        # directory already exists
        pass
