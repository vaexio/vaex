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
	
	def walk(self, f):
		return f(self)

class Scope(object):
	def __init__(self, name):
		self.name = name
		self.resolved = {}
	
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
	
		
if __name__ == "__main__":
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

	
	