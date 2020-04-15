from vaex.expresso import parse_expression, node_to_string, simplify, translate, validate_expression


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


def test_lists():
    node = parse_expression("searchsorted(['9E', 'AA', 'AQ', 'AS', 'B6', 'CO', 'DL', 'EV', 'F9', 'FL', 'HA', 'MQ', 'NW', 'OH', 'OO', 'UA', 'US', 'WN', 'XE', 'YV'], UniqueCarrier)")
    assert node is not None

    expr = "searchsorted(['9E', 'AA', 'AQ', 'AS', 'B6', 'CO', 'DL', 'EV', 'F9', 'FL', 'HA', 'MQ', 'NW', 'OH', 'OO', 'UA', 'US', 'WN', 'XE', 'YV'], UniqueCarrier)"
    expr_translate = translate(expr, lambda x: None)
    print(node)
    assert expr == expr_translate


def test_validate():
    validate_expression('x + 1', {'x'})
    validate_expression('x == "1"', {'x'})
