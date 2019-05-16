import vaex


def test_format():
    num1 = [1, 2, 3]
    num2 = [1.1, 2.2, 3.3]
    text = ['Here', 'we', 'go']

    df = vaex.from_arrays(num1=num1, num2=num2, text=text)

    assert df.num1.format("%d").tolist() == ['1', '2', '3']
    assert df.num1.format("%04d").tolist() == ['0001', '0002', '0003']
    assert df.num1.format('%f').tolist() == ['1.000000', '2.000000', '3.000000']
    assert df.num1.format('%05.2f').tolist() == ['01.00', '02.00', '03.00']
