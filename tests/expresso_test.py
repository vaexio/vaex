import pytest

from vaex.expresso import parse_expression, node_to_string, simplify, translate, validate_expression
from vaex.expresso import derivative, num
import vaex


@pytest.mark.parametrize('op', vaex.expression._binary_ops)
def test_binary_ops(op):
    expression = f'(a {op["code"]} b)'
    node = parse_expression(expression)
    assert node_to_string(node) == expression
    validate_expression(expression, {'a', 'b'})


@pytest.mark.parametrize('op', vaex.expression._unary_ops)
def test_unary_ops(op):
    expression = f'{op["code"]}a'
    node = parse_expression(expression)
    assert node_to_string(node) == expression


def test_compare():
    expr = "(x < 0)"
    node = parse_expression(expr)
    assert expr == node_to_string(node)

    expr = "((x < 0) >= 6)"
    node = parse_expression(expr)
    assert expr == node_to_string(node)

    expr = "(x // 10)"
    node = parse_expression(expr)
    assert expr == node_to_string(node)


def test_simplify():
    assert simplify("0 + 1") == "1"
    assert simplify("1 + 0") == "1"
    assert simplify("-0 + 1") == "1"
    assert simplify("1 + -0") == "1"

    assert simplify("0 * a + b") == "b"
    assert simplify("a * 0 + b") == "b"
    assert simplify("b + 0 * a") == "b"
    assert simplify("b + a * 0") == "b"


def test_kwargs():
    text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    df = vaex.from_arrays(text=text)
    expression = df.text.str.replace('[.]', '', regex=True)
    df.validate_expression(expression.expression)


def test_lists():
    node = parse_expression("searchsorted(['9E', 'AA', 'AQ', 'AS', 'B6', 'CO', 'DL', 'EV', 'F9', 'FL', 'HA', 'MQ', 'NW', 'OH', 'OO', 'UA', 'US', 'WN', 'XE', 'YV'], UniqueCarrier)")
    assert node is not None

    expr = "searchsorted(['9E', 'AA', 'AQ', 'AS', 'B6', 'CO', 'DL', 'EV', 'F9', 'FL', 'HA', 'MQ', 'NW', 'OH', 'OO', 'UA', 'US', 'WN', 'XE', 'YV'], UniqueCarrier)"
    validate_expression(expr, {'UniqueCarrier'}, {'searchsorted'})
    expr_translate = translate(expr, lambda x: None)
    print(node)
    assert expr == expr_translate

def test_dicts():
    expr = "fillmissing(o, {'a': 1, 'b': 2})"
    validate_expression(expr, {'o'}, {'fillmissing'})
    node = parse_expression(expr)
    assert node is not None
    expr_translate = translate(expr, lambda x: None)
    print(node)
    assert expr == expr_translate

def test_validate():
    validate_expression('x + 1', {'x'})
    validate_expression('x == "1"', {'x'})

def test_unicode():
    validate_expression('实体 + 1', {'实体'})


def test_num_returns_constant():
    import ast as _ast
    node = num(2)
    assert isinstance(node, _ast.Constant)
    assert node.value == 2


def test_derivative_of_power():
    # hits Derivative.visit_BinOp (Pow, `ast.Num` at line 325) and SimplifyExpression
    assert derivative("x**2", "x") == "(2 * x)"


def test_derivative_of_name():
    # hits Derivative.visit_Name (`ast.Num(n=1/0)` at lines 278/280)
    assert derivative("x", "x") == "1"
    assert derivative("y", "x") == "0"


def test_bool_not_folded():
    assert simplify("a * False") == "(a * False)"
    assert simplify("a * True") == "(a * True)"
    assert simplify("a * 0") == "0"
    assert simplify("a * 1") == "a"


def test_unary_printing():
    assert node_to_string(parse_expression("-x")) == "-x"
    assert node_to_string(parse_expression("~x")) == "~x"
    assert node_to_string(parse_expression("-(x+1)")) == "-((x + 1))"
    assert node_to_string(parse_expression("-'abc'")) == "-'abc'"
