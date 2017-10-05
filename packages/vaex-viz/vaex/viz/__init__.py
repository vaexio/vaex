import vaex.dataset
from vaex.utils import InnerNamespace

def add_namespace():
    vaex.dataset.Dataset.viz = InnerNamespace({})
    vaex.dataset.Dataset.viz._add(plot2d=vaex.dataset.Dataset.plot)
