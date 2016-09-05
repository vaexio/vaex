__author__ = 'maartenbreddels'
import unittest

def load_tests(loader, tests, pattern):
	import vaex.test.dataset
	import vaex.test.ui
	import vaex.test.misc
	print(loader, tests, pattern)
	for module in [vaex.test.dataset, vaex.test.ui, vaex.test.misc]:
		tests.addTests(unittest.TestLoader().loadTestsFromModule(module))
	return tests


