import yaml


class GlobalConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(GlobalConfig, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        with open('config/config.yaml', 'r') as file:
            cfg = yaml.safe_load(file)
        self.config = cfg

global_cfg = GlobalConfig()

