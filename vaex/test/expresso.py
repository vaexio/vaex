import vaex.dataset as dataset
import unittest
import vaex.expresso


class TestExpresso(unittest.TestCase):
	#def setUp(self):
	def test_ops(self):
		for binop in "+ - * / // % & | ^".split():
			vaex.expresso.validate_expression('a %s b' % binop, ['a', 'b'])
		for binop in "+ - * / // % & | ^".split():
			with self.assertRaises(NameError):
				vaex.expresso.validate_expression('a %s b' % binop, ['a'])
				vaex.expresso.validate_expression('a %s b' % binop, ['b'])
	def test_vars(self):
		with self.assertRaises(NameError):
			vaex.expresso.validate_expression('a+b', ['a'])
		vaex.expresso.validate_expression('a+b', ['a', 'b'])

	def test_functions(self):
		with self.assertRaises(NameError):
			vaex.expresso.validate_expression("f(a, b)", ['a'], ['f'])
		with self.assertRaises(NameError):
			vaex.expresso.validate_expression("f(a, b)", ['a', 'b'], ['g'])
		vaex.expresso.validate_expression("f(a, b)", ['a', 'b'], ['f'])

	def test_subsript(self):
		vaex.expresso.validate_expression("a[0]", ['a'])
		vaex.expresso.validate_expression("f(a[0])", ['a'], ['f'])
		vaex.expresso.validate_expression("a[0][0]", ['a'])
		with self.assertRaises(ValueError):
			vaex.expresso.validate_expression("a[0][b]", ['a', 'b'])
		with self.assertRaises(NameError):
			vaex.expresso.validate_expression("a[0]", [])



if __name__ == '__main__':
    unittest.main()
