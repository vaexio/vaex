import vaex.file

class ArrowOpener:
    @staticmethod
    def can_open(path, *args, **kwargs):
        return path.rpartition('.')[2] == 'arrow'

    @staticmethod
    def open(path, *args, **kwargs):
        from .dataset import DatasetArrow
        return DatasetArrow(path, *args, **kwargs)

def register_opener():
    vaex.file.register(ArrowOpener)
