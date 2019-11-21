import vaex.file

class ArrowOpener:
    @staticmethod
    def can_open(path, *args, **kwargs):
        return path.rpartition('.')[2] == 'arrow'

    @staticmethod
    def open(path, *args, **kwargs):
        from .dataset import open
        return open(path, *args, **kwargs)

class ParquetOpener:
    @staticmethod
    def can_open(path, *args, **kwargs):
        return path.rpartition('.')[2] == 'parquet'

    @staticmethod
    def open(path, *args, **kwargs):
        from .dataset import open_parquet
        return open_parquet(path, *args, **kwargs)

def register_opener():
    vaex.file.register(ArrowOpener)
    vaex.file.register(ParquetOpener)
