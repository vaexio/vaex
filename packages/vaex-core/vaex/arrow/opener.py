import vaex.file

class ArrowOpener:
    @classmethod
    def quick_test(cls, path, *args, **kwargs):
        return path.rpartition('.')[2] == 'arrow'

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        return cls.quick_test(path, *args, **kwargs)

    @staticmethod
    def open(path, *args, **kwargs):
        from .dataset import open
        return open(path, *args, **kwargs)

class ParquetOpener:
    @classmethod
    def quick_test(cls, path, *args, **kwargs):
        return path.rpartition('.')[2] == 'parquet'

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        return cls.quick_test(path, *args, **kwargs)

    @staticmethod
    def open(path, *args, **kwargs):
        from .dataset import open_parquet
        return open_parquet(path, *args, **kwargs)
