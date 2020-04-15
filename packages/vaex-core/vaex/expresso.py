# -*- coding: utf-8 -*-
from __future__ import division
import logging
import collections
import ast
import _ast
import string
import numpy as np
import math
import sys
import six
import copy
import difflib


if hasattr(_ast, 'Num'):
    ast_Num = _ast.Num
    ast_Str = _ast.Str
else:  # Python3.8
    ast_Num = _ast.Constant
    ast_Str = _ast.Constant


logger = logging.getLogger("expr")
logger.setLevel(logging.ERROR)


valid_binary_operators = [_ast.Add, _ast.Sub, _ast.Mult, _ast.Pow,
                          _ast.Div, _ast.FloorDiv, _ast.BitAnd, _ast.BitOr, _ast.BitXor, _ast.Mod]
valid_compare_operators = [_ast.Lt, _ast.LtE,
                           _ast.Gt, _ast.GtE, _ast.Eq, _ast.NotEq]
valid_unary_operators = [_ast.USub, _ast.UAdd, _ast.Invert]
valid_id_characters = string.ascii_letters + string.digits + "_"
valid_functions = "sin cos".split()

opmap = {
    _ast.Add: '+',
    _ast.Sub: '-',
    _ast.Mult: '*',
    _ast.Pow: '**',
    _ast.Div: '/',
    _ast.FloorDiv: '//',
    _ast.BitAnd: '&',
    _ast.BitOr: '|',
    _ast.BitXor: '^',
    _ast.Mod: '%',
}


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


last_func = None


def validate_expression(expr, variable_set, function_set=[], names=None):
    global last_func
    names = names if names is not None else []
    if isinstance(expr, six.string_types):
        node = ast.parse(expr)
        if len(node.body) != 1:
            raise ValueError("expected one expression, got %r" %
                             len(node.body))
        first_expr = node.body[0]
        if not isinstance(first_expr, _ast.Expr):
            raise ValueError("expected an expression got a %r" %
                             type(node.body))
        validate_expression(first_expr.value, variable_set,
                            function_set, names)
    elif isinstance(expr, _ast.BinOp):
        if expr.op.__class__ in valid_binary_operators:
            validate_expression(expr.right, variable_set, function_set, names)
            validate_expression(expr.left, variable_set, function_set, names)
        else:
            raise ValueError("Binary operator not allowed: %r" % expr.op)
    elif isinstance(expr, _ast.UnaryOp):
        if expr.op.__class__ in valid_unary_operators:
            validate_expression(expr.operand, variable_set,
                                function_set, names)
        else:
            raise ValueError("Unary operator not allowed: %r" % expr.op)
    elif isinstance(expr, _ast.Name):
        validate_id(expr.id)
        if expr.id not in variable_set:
            matches = difflib.get_close_matches(expr.id, list(variable_set))
            msg = "Column or variable %r does not exist." % expr.id
            if matches:
                msg += ' Did you mean: ' + " or ".join(map(repr, matches))

            raise NameError(msg)
        names.append(expr.id)
    elif isinstance(expr, ast_Num):
        pass  # numbers are fine
    elif isinstance(expr, ast_Str):
        pass  # as well as strings
    elif isinstance(expr, _ast.Call):
        validate_func(expr.func, function_set)
        last_func = expr
        for arg in expr.args:
            validate_expression(arg, variable_set, function_set, names)
        for arg in expr.keywords:
            validate_expression(arg, variable_set, function_set, names)
    elif isinstance(expr, _ast.Compare):
        validate_expression(expr.left, variable_set, function_set, names)
        for op in expr.ops:
            if op.__class__ not in valid_compare_operators:
                raise ValueError("Compare operator not allowed: %r" % op)
        for comparator in expr.comparators:
            validate_expression(comparator, variable_set, function_set, names)
    elif isinstance(expr, _ast.keyword):
        validate_expression(expr.value, variable_set, function_set, names)
    elif isinstance(expr, _ast.Subscript):
        validate_expression(expr.value, variable_set, function_set, names)
        if isinstance(expr.slice.value, _ast.Num):
            pass  # numbers are fine
        else:
            raise ValueError(
                "Only subscript/slices with numbers allowed, not: %r" % expr.slice.value)
    else:
        last_func = expr
        raise ValueError("Unknown expression type: %r" % type(expr))


class Validator(ast.NodeVisitor):

    def generic_visit(self, node):
        raise ValueError('unexpected node: {}', ast.dump(node))

    def visit_BinOp(self, expr):
        if expr.op.__class__ in valid_binary_operators:
            validate_expression(expr.right, variable_set, function_set, names)
            validate_expression(expr.left, variable_set, function_set, names)
        else:
            raise ValueError("Binary operator not allowed: %r" % expr.op)


def mul(left, right):
    return ast.BinOp(left=left, right=right, op=ast.Mult())


def div(left, right):
    return ast.BinOp(left=left, right=right, op=ast.Div())


def add(left, right):
    return ast.BinOp(left=left, right=right, op=ast.Add())


def sub(left, right):
    return ast.BinOp(left=left, right=right, op=ast.Sub())


def pow(left, right):
    return ast.BinOp(left=left, right=right, op=ast.Pow())


def sqr(node):
    return ast.BinOp(left=node, right=num(2), op=ast.Pow())

def sqrt(node):
    return call('sqrt', [node])


def neg(node):
    return ast.UnaryOp(op=ast.USub(), operand=node)


def num(n):
    return ast.Num(n=n)


def call(fname, args):
    return ast.Call(func=ast.Name(id=fname, ctx=ast.Load()), args=args)


def _dlog10(n, args):
    assert len(args) == 1
    assert n == 0
    a = call('log', args=[num(10)])
    return div(num(1), mul(args[0], a))


def _dsqrt(n, args):
    assert n == 0
    assert len(args) == 1
    a = call('log', args=[num(10)])
    return mul(num(1/2), pow(args[0], num(-0.5)))


def _dcos(n, args):
    assert n == 0
    assert len(args) == 1
    return neg(call('sin', args=args))

def _darccos(n, args):
    assert n == 0
    assert len(args) == 1
    a = sqrt(sub(num(1), sqr(args[0])))
    return neg(div(num(1), a))

def _darctan2(n, args):
    # derivative of arctan2(y, x)
    assert (n >= 0) and (n <= 1)
    assert len(args) == 2
    y, x = args
    if n == 1:  # derivative wrt 2nd argument (x)
        return div(neg(y), add(sqr(x), sqr(y)))
    if n == 0:  # derivative wrt 1st argument (y)
        return div(x, add(sqr(x), sqr(y)))

def _dtan(n, args):
    assert n == 0
    assert len(args) == 1
#     a = div(sub(num(1), sqr(args[0])))
    return div(num(1), sqr(call('cos', args=args)))

standard_function_derivatives = {}
standard_function_derivatives['sin'] = 'cos'
standard_function_derivatives['cos'] = _dcos
standard_function_derivatives['tan'] = _dtan
standard_function_derivatives['log10'] = _dlog10
standard_function_derivatives['sqrt'] = _dsqrt
standard_function_derivatives['arctan2'] = _darctan2
standard_function_derivatives['arccos'] = _darccos


class Derivative(ast.NodeTransformer):
    def __init__(self, id, function_derivatives={}):
        self.id = id
        self.function_derivatives = dict(standard_function_derivatives)
        self.function_derivatives.update(function_derivatives)

    def format(self, node):
        # try:
        return ExpressionString().visit(node)
        # return ast.dump(node)

    def visit_Num(self, node):
        return ast.Num(n=0)

    def visit_Name(self, node):
        if node.id == self.id:
            return ast.Num(n=1)
        else:
            return ast.Num(n=0)

    def visit_Call(self, node):
        fname = node.func.id
        df = self.function_derivatives.get(fname)
        if df is None:
            raise ValueError('Derivative of {} is unknown'.format(fname))
        if not callable(df):  # simply a string
            assert len(node.args) == 1
            result = mul(call(df, node.args), self.visit(node.args[0]))
        else:
            terms = [mul(df(i, node.args), self.visit(arg))
                     for i, arg in enumerate(node.args)]
            result = terms[0]
            for term in terms[1:]:
                result = add(result, term)
        return result

    def generic_visit(self, node):
        # it's annoying that the default one modifies in place
        return super(Derivative, self).generic_visit(copy.deepcopy(node))

    def visit_BinOp(self, node):
        solution = None
        if isinstance(node.op, ast.Mult):
            solution = add(mul(self.visit(node.left), node.right),
                           mul(node.left,             self.visit(node.right)))
        if isinstance(node.op, ast.Div):
            # (n*at - t*an) / n2
            n = node.right
            t = node.left
            at = self.visit(t)
            an = self.visit(n)
            solution = div(sub(mul(n, at), mul(t, an)), pow(n, num(2)))
        if isinstance(node.op, ast.Add):
            solution = add(self.visit(node.left), self.visit(node.right))
        if isinstance(node.op, ast.Sub):
            solution = sub(self.visit(node.left), self.visit(node.right))
        if isinstance(node.op, ast.Pow):
            # following https://en.wikipedia.org/wiki/Differentiation_rules
            f = node.left
            df = self.visit(f)
            g = node.right
            dg = self.visit(g)
            # if g is a number, we take a equivalent solution, which gives a nicer result
            if isinstance(g, ast.Num):
                solution = mul(g, mul(df, pow(node.left, num(node.right.n-1))))
            else:
                a = add(mul(df, div(g, f)), mul(dg, call('log', [f])))
                solution = mul(pow(f, g), a)
        if solution is None:
            raise ValueError('Unknown rule for: {}'.format(self.format(node)))
        return solution


class ExpressionString(ast.NodeVisitor):
    def __init__(self, pretty=False):
        self.pretty = pretty
        self.indent = 0
    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub):
            if isinstance(node.operand, (ast.Name, ast.Num)):
                return "-{}".format(self.visit(node.operand))  # prettier
            else:
                return "-({})".format(self.visit(node.operand))
        if isinstance(node.op, ast.Invert):
            if isinstance(node.operand, (ast.Name, ast.Num)):
                return "~{}".format(self.visit(node.operand))  # prettier
            else:
                return "~({})".format(self.visit(node.operand))
        # elif isinstance(node.op, ast.UAdd):
        #     return "{}".format(self.visit(self.operatand))
        else:
            raise ValueError('Unary op not supported: {}'.format(node.op))

    def visit_Name(self, node):
        return node.id

    def visit_Num(self, node):
        return repr(node.n)

    def visit_keyword(self, node):
        return "%s=%s" % (node.arg, self.visit(node.value))

    def visit_NameConstant(self, node):
        return repr(node.value)

    def visit_Call(self, node):
        args = [self.visit(k) for k in node.args]
        keywords = []
        if hasattr(node, 'keywords'):
            keywords = [self.visit(k) for k in node.keywords]
        return "{}({})".format(node.func.id, ", ".join(args + keywords))

    def visit_Str(self, node):
        return repr(node.s)

    def visit_List(self, node):
        return "[{}]".format(", ".join([self.visit(k) for k in node.elts]))

    def visit_BinOp(self, node):
        newline = indent = ""
        if self.pretty:
            indent = "  " * self.indent
            newline = "\n"
        self.indent += 1
        left = "{}{}{}".format(newline, indent, self.visit(node.left))
        right = "{}{}{}".format(newline, indent, self.visit(node.right))
        try:
            if isinstance(node.op, ast.Mult):
                return "({left} * {right})".format(left=left, right=right)
            if isinstance(node.op, ast.Div):
                return "({left} / {right})".format(left=left, right=right)
            if isinstance(node.op, ast.FloorDiv):
                return "({left} // {right})".format(left=left, right=right)
            if isinstance(node.op, ast.Add):
                return "({left} + {right})".format(left=left, right=right)
            if isinstance(node.op, ast.Sub):
                return "({left} - {right})".format(left=left, right=right)
            if isinstance(node.op, ast.Pow):
                return "({left} ** {right})".format(left=left, right=right)
            if isinstance(node.op, ast.BitAnd):
                return "({left} & {right})".format(left=left, right=right)
            if isinstance(node.op, ast.BitOr):
                return "({left} | {right})".format(left=left, right=right)
            if isinstance(node.op, ast.BitXor):
                return "({left} ^ {right})".format(left=left, right=right)
            else:
                return "do_not_understand_expression"
        finally:
            self.indent -= 1

    op_translate = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
    def visit_Compare(self, node):
        s = ""
        left = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = self.visit(comp)
            op = ExpressionString.op_translate[op.__class__]
            s = "({left} {op} {right})".format(left=left, op=op, right=right)
            left = right
        return s


class SimplifyExpression(ast.NodeTransformer):

    def visit_UnaryOp(self, node):
        node.operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Num) and node.operand.n == 0:
                node = node.operand
        return node

    def visit_BinOp(self, node):
        node.left = left = self.visit(node.left)
        node.right = right = self.visit(node.right)
        if isinstance(node.op, ast.Mult):
            if isinstance(right, ast.Num) and right.n == 0:
                return num(0)
            elif isinstance(right, ast.Num) and right.n == 1:
                return left
            elif isinstance(left, ast.Num) and left.n == 0:
                return num(0)
            elif isinstance(left, ast.Num) and left.n == 1:
                return right
        if isinstance(node.op, ast.Div):
            if isinstance(left, ast.Num) and left.n == 0:
                return num(0)
        if isinstance(node.op, ast.Add):
            if isinstance(right, ast.Num) and right.n == 0:
                return left
            if isinstance(left, ast.Num) and left.n == 0:
                return right
        if isinstance(node.op, ast.Sub):
            if isinstance(right, ast.Num) and right.n == 0:
                return left
            if isinstance(left, ast.Num) and left.n == 0:
                return neg(right)
        if isinstance(node.op, ast.Pow):
            if isinstance(left, ast.Num) and left.n == 0:
                return num(0)  # not ok with negative powers..
            if isinstance(right, ast.Num) and right.n == 0:
                # TODO: this means a numpy arrays can become a scalar
                return num(1)
            if isinstance(right, ast.Num) and right.n == 1:
                return left
        return node


class Translator(ast.NodeTransformer):
    def __init__(self, translator):
        self.translator = translator

    def visit_Call(self, node):
        # we skip visiting node.id
        node.args = [self.visit(k) for k in node.args]
        if hasattr(node, 'keywords'):
            node.keywords = [self.visit(k) for k in node.keywords]
        return node

    def visit_Name(self, node):
        expr = self.translator(node.id)
        if expr:
            node = parse_expression(expr)
            node = self.visit(node)
        return node


class NameCollector(ast.NodeTransformer):
    def __init__(self):
        self.names = {}

    def visit_Call(self, node):
        # we skip visiting node.id
        node.args = [self.visit(k) for k in node.args]
        if hasattr(node, 'keywords'):
            node.keywords = [self.visit(k) for k in node.keywords]
        return node

    def visit_Name(self, node):
        if node.id not in self.names:
            self.names[node.id] = []
        self.names[node.id].append(node)
        return node

class GraphBuiler(ast.NodeVisitor):
    def __init__(self):
        self.dependencies = []

    def visit_Call(self, node):
        fname = node.func.id
        dependencies = list(self.dependencies)
        self.dependencies = []
        for arg in node.args:
            self.visit(arg)
        graph = [fname, node_to_string(node), self.dependencies]
        dependencies.append(graph)
        self.dependencies = dependencies

    def visit_BinOp(self, node):
        dependencies = list(self.dependencies)
        self.dependencies = []
        self.visit(node.left)
        dep_left = self.dependencies

        self.dependencies = []
        self.visit(node.right)
        dep_right = self.dependencies
        graph = [opmap[type(node.op)], node_to_string(node), dep_left + dep_right]
        dependencies.append(graph)
        self.dependencies = dependencies

    def visit_Name(self, node):
        self.dependencies.append(node.id)


def _graph(expression_string):
    node = parse_expression(expression_string)
    g = GraphBuiler()
    node = g.visit(node)
    return g.dependencies[0]


def simplify(expression_string):
    node = parse_expression(expression_string)
    node = SimplifyExpression().visit(node)
    return node_to_string(node)


def derivative(expression, variable_name, simplify=True):
    if isinstance(expression, str):
        node = parse_expression(expression)
    else:
        node = expression
    node = Derivative(variable_name).visit(node)
    if simplify:
        node = SimplifyExpression().visit(node)
    return node_to_string(node)


def translate(expression, translator):
    if isinstance(expression, str):
        node = parse_expression(expression)
    else:
        node = expression
    node = Translator(translator).visit(node)
    return node_to_string(node)


def names(expression):
    if isinstance(expression, str):
        node = parse_expression(expression)
    else:
        node = expression
    nc = NameCollector()
    nc.visit(node)
    return nc.names


def parse_expression(expression_string):
    expr = ast.parse(expression_string).body[0]
    assert isinstance(expr, ast.Expr), "not an expression"
    return expr.value


def node_to_string(node, pretty=False):
    return ExpressionString(pretty=pretty).visit(node)


def validate_func(name, function_set):
    if name.id not in function_set:
        raise NameError("function %r is not defined" % name.id)


def validate_id(id):
    for char in id:
        if char not in valid_id_characters:
            raise ValueError("invalid character %r in id %r" % (char, id))
