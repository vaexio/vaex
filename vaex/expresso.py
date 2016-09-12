# -*- coding: utf-8 -*-
import logging
import logging
import collections

logger = logging.getLogger("expr")
logger.setLevel(logging.ERROR)

class Base(object):
	def __neg__(self):
		return Neg(self)
	def __add__(self, rhs):
		return Add(self, rhs)
	def __radd__(self, lhs):
		return Add(lhs, self)
	def __sub__(self, rhs):
		return Sub(self, rhs)
	def __rsub__(self, lhs):
		return Sub(lhs, self)
	def __mul__(self, rhs):
		return Mul(self, rhs)
	def __rmul__(self, lhs):
		return Mul(lhs, self)
	def __div__(self, rhs):
		return Div(self, rhs)
	def __pow__(self, rhs):
		return Pow(self, rhs)

	def walk(self, f):
		return f(self)

class Sqrt(Base):
	def __init__(self, value):
		self.value = check_const(value)

	def __repr__(self):
		return "sqrt(" + repr(self.value) + ")"
class Const(Base):
	def __init__(self, value):
		self.value = value

	def __repr__(self):
		return repr(self.value)

	def c_code(self):
		return repr(self.value)

def check_const(value):
	if isinstance(value, (int, float)):
		return Const(value)
	else:
		return value

class Neg(Base):
	def __init__(self, obj):
		self.obj = check_const(obj)

	def __repr__(self):
		return "(-" +repr(self.obj) + ")"

	def walk(self, f):
		#logger.debug("walk neg")
		return f(Neg(self.obj.walk(f)))

class Add(Base):
	def __init__(self, lhs, rhs):
		self.lhs = check_const(lhs)
		self.rhs = check_const(rhs)

	def __repr__(self):
		return "(" +repr(self.lhs) +" + " +repr(self.rhs) +")"

	def walk(self, f):
		return f(Add(self.lhs.walk(f), self.rhs.walk(f)))

	def c_code(self):
		return "(" +self.lhs.c_code() +" + " +self.rhs.c_code() +")"

class Sub(Base):
	def __init__(self, lhs, rhs):
		self.lhs = check_const(lhs)
		self.rhs = check_const(rhs)

	def __repr__(self):
		return "(" +repr(self.lhs) +" - " +repr(self.rhs) +")"

	def walk(self, f):
		return f(Sub(self.lhs.walk(f), self.rhs.walk(f)))

class Div(Base):
	def __init__(self, lhs, rhs):
		self.lhs = check_const(lhs)
		self.rhs = check_const(rhs)

	def __repr__(self):
		return "(" +repr(self.lhs) +" / " +repr(self.rhs) +")"

	def walk(self, f):
		return f(Div(self.lhs.walk(f), self.rhs.walk(f)))

class Mul(Base):
	def __init__(self, lhs, rhs):
		self.lhs = check_const(lhs)
		self.rhs = check_const(rhs)

	def __repr__(self):
		return "(" +repr(self.lhs) +" * " +repr(self.rhs) +")"

	def c_code(self):
		return "(" +self.lhs.c_code() +" * " +self.rhs.c_code() +")"

	def walk(self, f):
		return f(Mul(self.lhs.walk(f), self.rhs.walk(f)))

class Pow(Base):
	def __init__(self, lhs, rhs):
		self.lhs = check_const(lhs)
		self.rhs = check_const(rhs)

	def __repr__(self):
		return "(" +repr(self.lhs) +" ** " +repr(self.rhs) +")"

	def walk(self, f):
		return f(Pow(self.lhs.walk(f), self.rhs.walk(f)))

class Function(Base):
	def __init__(self, var, args):
		self.var = var
		self.args = args
		#print args

	def __repr__(self):
		return self.var.name +"(" +", ".join([repr(k) for k in self.args]) + ")"

	def walk(self, f):
		args = [arg.walk(f) for arg in self.args]
		return f(Function(self.var.walk(f), args))


class Slice(Base):
	def __init__(self, var, args):
		self.var = var
		self.args = args
		try:
			len(self.args)
		except:
			self.args = (self.args,)

	def tovar(self):
		parts = [self.var.name]
		for arg in self.args:
			if isinstance(arg, (int, float)):
				parts.append(repr(arg).replace("-", "_min_"))
			elif isinstance(arg, Var):
				parts.append(arg.name)
			elif arg is None:
				parts.append("None")
			else:
				s = ""
				s += repr(arg.start)
				s += "_"
				s += repr(arg.stop)
				s += "_"
				s += repr(arg.step)
				parts.append(s)
		return "__".join(parts)


	def __repr__(self):
		parts = []
		#logger.debug("args: %r" % self.args)
		for arg in self.args:
			if isinstance(arg, (int, float)):
				parts.append(repr(arg))
			elif isinstance(arg, Var):
				parts.append(arg.name)
			elif arg is None:
				parts.append("")
			else:
				s = ""
				if arg.start is not None:
					s += repr(arg.start)
				s += ":"
				if arg.stop is not None:
					s += repr(arg.stop)
				s += ":"
				if arg.step is not None:
					s += repr(arg.step)
				if s == "::":
					s = ":"
				parts.append(s)
		return self.var.name + "[" +",".join(parts) + "]"

class Var(Base):
	def __init__(self, name):
		self.name = name

	def __getitem__(self, args):
		logger.debug("[%s] Var.__getitem__(%r)" % (self.name, args,))
		return Slice(self, args)

	def __call__(self, *args):
		logger.debug("[%s] Var.__call__(%r)" % (self.name, args,))
		return Function(self, args)

	def __str__(self):
		return "Var(%r)" % self.name
	def __repr__(self):
		return self.name

	def c_code(self):
		return self.name

	def walk(self, f):
		return f(self)

class Scope(object):
	def __init__(self, name):
		self.name = name
		self.resolved = collections.OrderedDict()

	def write(self, f):
		for block_index, (key, value) in enumerate(self.resolved.items()):
			# this assumes the var is part of a 'block'
			f.write("double %s = blocks[%i][index];\n" % (key, block_index) )

	def __getitem__(self, name):
		logger.debug("[%s] Scope.__getitem__(%r)" % (self.name, name,))
		if name not in self.resolved:
			self.resolved[name] = Var(name)
		else:
			logger.debug("[%s] Scope.__getitem__(%r) (resolved again)" % (self.name, name,))
		return self.resolved[name]

		#return Scope(name)

def distance(args):
	if len(args) == 1:
		result = args[0]
	else:
		result = Add(args[0]**2, args[1]**2)
		for arg in args[2:]:
			result = Add(result, arg**2)
		result = Sqrt(result)
	return result
macros = {"l2": distance}
def translate(expression, replacements={}):
	scope = Scope(None)
	newvars = {}
	used_vars = set()
	result = eval(expression, {}, scope)
	#for key, value in replacements:
	translated_replacements = {}
	for i, (key, expression) in enumerate(replacements.items()):
		current_replacements = collections.OrderedDict(list(replacements.items())[:i])
		#print "*" * 70
		translated_replacements[key] = translate(expression, current_replacements)[0]
		#print "TRANSLATE:", expression, translated_replacements[key], current_replacements

	def slice_to_var(slice):
		newname = slice.tovar()
		logger.debug("replace %s by %s" % (slice.var.name, newname))
		newvars[newname] = slice
		return Var(newname)

	def walker(expr):
		#print ">>>>", expr, type(expr)
		if isinstance(expr, Slice):
			return slice_to_var(expr)
		elif isinstance(expr, Function):
			#logger.debug("expr: " + expr.var.name)
			if expr.var.name in macros:
				return macros[expr.var.name](expr.args)
			else:
				return expr
		elif isinstance(expr, Var):
			#logger.debug("var: " + expr.name)
			used_vars.add(expr.name)
			if expr.name in translated_replacements:
				return translated_replacements[expr.name]
			else:
				return expr
		else:
			return expr
	#print "walking.."
	newresult = result.walk(walker)
	return newresult, newvars #, used_vars

class MathExpression(object):
	def __init__(self, expression_string, macros):
		self.expression_string = expression_string
		self.scope = Scope()
		self.macros = macros
		for macro in self.macros:
			scope.add_macro(macro)
		self.expression = eval(self.expression_string)

	def evaluate(self, values):
		return eval(self.expression, math.__dict__, values)

	def evaluate_numpy(self, values):
		return eval(self.expression, np.__dict__, values)

	def variables(self):
		eval(self.expression, np.__dict__, scope)

import ast
import _ast
import string
import numpy as np
import math
import sys

valid_binary_operators = [_ast.Add, _ast.Sub, _ast.Mult, _ast.Pow, _ast.Div, _ast.BitAnd, _ast.BitOr, _ast.BitXor, _ast.Mod]
valid_compare_operators = [_ast.Lt, _ast.LtE, _ast.Gt, _ast.GtE, _ast.Eq, _ast.NotEq]
valid_unary_operators = [_ast.USub, _ast.UAdd]
valid_id_characters = string.ascii_letters + string.digits + "_"
valid_functions = "sin cos".split()

def math_parse(expression, macros=[]):
	# TODO: validate macros?
	node = ast.parse(expression)
	if len(node.body) != 1:
		raise ValueError("expected one expression, got %r" % len(node.body))
	expr = node.body[0]
	if not isinstance(expr, _ast.Expr):
		raise ValueError("expected an expression got a %r" % type(node.body))
	validate_expression(expr.value)
	return MathExpression(expression, macros)
import six
def validate_expression(expr, variable_set, function_set):
	if isinstance(expr, six.string_types):
		node = ast.parse(expr)
		if len(node.body) != 1:
			raise ValueError("expected one expression, got %r" % len(node.body))
		first_expr = node.body[0]
		if not isinstance(first_expr, _ast.Expr):
			raise ValueError("expected an expression got a %r" % type(node.body))
		validate_expression(first_expr.value, variable_set, function_set)
	elif isinstance(expr, _ast.BinOp):
		if expr.op.__class__ in valid_binary_operators:
			validate_expression(expr.right, variable_set, function_set)
			validate_expression(expr.left, variable_set, function_set)
		else:
			raise ValueError("Binary operator not allowed: %r" % expr.op)
	elif isinstance(expr, _ast.UnaryOp):
		if expr.op.__class__ in valid_unary_operators:
			validate_expression(expr.operand, variable_set, function_set)
		else:
			raise ValueError("Unary operator not allowed: %r" % expr.op)
	elif isinstance(expr, _ast.Name):
		validate_id(expr.id)
		if expr.id not in variable_set:
			raise NameError("variable %r is not defined (available are: %s]" % (expr.id, ", ".join(list(variable_set))))
	elif isinstance(expr, _ast.Num):
		pass # numbers are fine
	elif isinstance(expr, _ast.Call):
		validate_func(expr.func, function_set)
	elif isinstance(expr, _ast.Compare):
		validate_expression(expr.left, variable_set, function_set)
		for op in expr.ops:
			if op.__class__ not in valid_compare_operators:
				raise ValueError("Compare operator not allowed: %r" % op)
		for comparator in expr.comparators:
			validate_expression(comparator, variable_set, function_set)
		#if expr.op.__class__ in valid_binary_operators:
		#import pdb
		#pdb.set_trace()
	else:
		raise ValueError("Unknown expression type: %r" % type(expr))


def validate_func(name, function_set):
	if name.id not in function_set:
		raise NameError("function %r is not defined" % name.id)
	#if name.id not in valid_functions:
	#	raise ValueError, "invalid or unknown function %r" % name.id


def validate_id(id):
	for char in id:
		if char not in valid_id_characters:
			raise ValueError("invalid character %r in id %r" % (char, id))

expressions = [expr for expr in """
a * b
a + b
sin(a)
a**b - sin(a**2, b--a)
""".split("\n") if expr]

if __name__ == "__main__":
	values = dict(a=1, b=2, pi=math.pi)
	for expression in expressions:
		ex = math_parse(expression)
		print((expression, "=", ex.evaluate(values)))
	values = dict(a=np.array([1,2]), b=np.array([2,3]), pi=math.pi)
	for expression in expressions:
		ex = math_parse(expression)
		print(expression, "=", ex.evaluate_numpy(values))
	import types
	import numba

	numba.jit
	types.FunctionType()
	sys.exit(0)
	import vaex as vx
	from setuptools import setup, Extension
	ds = vx.example()
	scope = Scope(ds)
	result = eval("x + y*3", {}, scope)
	import StringIO
	f = StringIO.StringIO()
	f.write("// this is generated code =====\n")
	scope.write(f)
	f.write("values[0] = %s;" % result.c_code())
	f.write("// end of generated code =====\n")
	print(f.getvalue())
	import os
	import platform
	from distutils.sysconfig import get_python_inc

	base_path = os.path.expanduser("~/.vaex/ext")
	#build_path = os.path.expanduser("~/.vaex/ext/build")
	#build_path = os.path.expanduser("~/.vaex/ext/temp")
	if not os.path.exists(base_path):
		os.makedirs(base_path)

	filename = os.path.join(base_path, "code.cpp")
	template = file(os.path.expanduser("src/vaex/vaexfast_template.cpp")).read()
	with open(filename, "w") as file:
		file.write(template.replace("{code_nd}", f.getvalue()))
	try:
		import numpy
		numdir = os.path.dirname(numpy.__file__)
	except:
		numdir = None

	if numdir is None:
		print("numpy not found, cannot install")

	include_dirs = []
	library_dirs = []
	libraries = []
	defines = []
	if "darwin" in platform.system().lower():
		extra_compile_args = ["-mfpmath=sse", "-O3", "-funroll-loops"]
	else:
		#extra_compile_args = ["-mfpmath=sse", "-msse4", "-Ofast", "-flto", "-march=native", "-funroll-loops"]
		extra_compile_args = ["-mfpmath=sse", "-msse4", "-Ofast", "-flto", "-funroll-loops"]
		#extra_compile_args = ["-mfpmath=sse", "-O3", "-funroll-loops"]
		extra_compile_args = ["-mfpmath=sse", "-msse4a", "-O3", "-funroll-loops"]
	extra_compile_args.extend(["-std=c++0x"])

	include_dirs.append(os.path.join(get_python_inc(plat_specific=1), "numpy"))
	if numdir is not None:
		include_dirs.append(os.path.join(numdir, "core", "include"))

	ex = Extension("vaexfast_user", [filename],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                define_macros=defines,
                extra_compile_args=extra_compile_args
                )
	from setuptools import setup, Extension
	setup(script_args=["build_ext", "--build-lib=" + base_path], ext_modules=[ex])



	if 0:
		logger.setLevel(logging.DEBUG)

		expr1 = "log(a[:,index]) + index"
		expr1 = "log(a[index]) + index"
		expr1 = "log(a[-1]) - index + 1 + 1**2 + index**index + d(a,b) + 1.2"
		expr1 = "-2.5 * log10(SPECTROFLUX[:,2])"
		scope = Scope(None)
		result = eval(expr1, {}, scope)
		print(("expression", expr1))
		print(("should equal", repr(result)))
		print("translating")
		print()
		newexpr, vars = translate(expr1)
		print((newexpr, vars))

		expr = "xc + yc"
		newexpr, vars = translate(expr, {"xc":"x+1", "yc":"sqrt(x**2+y**2)"})
		print((newexpr, vars))


		expr = "(xc + yc)/r"
		replacements = collections.OrderedDict()
		replacements["xc"] = "x+1"
		replacements["yc"] = "y-1"
		replacements["r"] = "sqrt(xc**2+yc**2)"
		print((list(replacements.items())))
		newexpr, vars = translate(expr, replacements)
		print((newexpr, vars))




