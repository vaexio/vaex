__author__ = 'breddels'
import logging
logger = logging.getLogger("vaex.file")

import vaex.file.other

def can_open(path, *args, **kwargs):
	for name, class_ in list(vaex.file.other.dataset_type_map.items()):
		if class_.can_open(path, *args):
			return True


def open(path, *args, **kwargs):
	dataset_class = None
	for name, class_ in list(vaex.file.other.dataset_type_map.items()):
		logger.debug("trying %r with class %r" % (path, class_))
		if class_.can_open(path, *args, **kwargs):
			logger.debug("can open!")
			dataset_class = class_
			break
	if dataset_class:
		dataset = dataset_class(path, *args)
		return dataset
