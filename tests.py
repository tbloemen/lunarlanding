import pytest
from compare_expressions import compare_multitrees, convert_ty_sympy
from sympy import sympify

def test_convert_ty_sympy_positive_and_negative_constants():
    expr_str = "x_1 + 3 - 5.5 + x_2 - -2"
    expr = convert_ty_sympy(expr_str)
    expected = sympify("x_1 + 1 - 1 + x_2 - -1")
    assert expr.equals(expected)

def test_convert_ty_sympy_no_constants():
    expr_str = "x_0 * x_1 + x_2"
    expr = convert_ty_sympy(expr_str)
    assert str(expr) == "x_0*x_1 + x_2"

def test_compare_multitrees_equal():
    t1 = ["x_0 + 1", "x_1 - 2", "x_2 * 3", "x_3 / -4"]
    t2 = ["x_0 + 5", "x_1 - 7", "x_2 * 9", "x_3 / -8"]
    assert compare_multitrees(t1, t2) == 4

def test_compare_multitrees_partial():
    t1 = ["x_0 + 1", "x_1 - 2", "x_2 * 3", "x_3 / -4"]
    t2 = ["x_0 + 1", "x_1 - 7", "x_2 * 9", "x_3 / 8"]
    assert compare_multitrees(t1, t2) == 3

def test_compare_multitrees_none():
    t1 = ["x_0 + 1", "x_1 - 2", "x_2 * 3", "x_3 / -4"]
    t2 = ["y_0 + 1", "y_1 - 2", "y_2 * 3", "y_3 / -4"]
    assert compare_multitrees(t1, t2) == 0