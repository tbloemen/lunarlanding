import pytest
from compare_expressions import *
from sympy import sympify

def test_convert_ty_sympy_positive_and_negative_constants():
    expr_str = "x_1 + 3 - 5.5 + x_2 - -2"
    expr = convert_to_sympy_ones(expr_str)
    expected = sympify("x_1 + 1 - 1 + x_2 - -1")
    assert expr.equals(expected)
    e = '-1'
    assert sympify(e).equals(convert_to_sympy_ones(e))

def test_convert_ty_sympy_no_constants():
    expr_str = "x_0 * x_1 + x_2"
    expr = convert_to_sympy_ones(expr_str)
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

def test_it_failed_once():
    s = '((((x_1+x_1)*0)-(x_1-x_5))-(x_5/((x_0+0)/x_5)))'
    e = convert_to_sympy_ones(s)
    assert 0 == 0

def test_it_failed_too():
    s = "((x_3+(x_3+x_4))/(((x_4+x_2)/5)/(-2*(((x_1*(x_6-x_2))*x_6)0.3))))"
    s2 = "((x_3+(x_3+x_4))/(((x_4+x_2)/5)/(-2*(((x_1*(x_6-x_2))*x_6)*0.1))))"
    o = convert_to_sympy_round(s)
    o2 = convert_to_sympy_round(s2)
    print(o)
    assert o.equals(o2)

def this_failed_as_well():
    s = '((x_3-((x_2-0.2)*x_2))*(x_2-((x_4-2)-x_5)))'
    s2 = '((x_3-((x_2-0.1)*x_2))*(x_2-((x_4-2)-x_5)))'
    o = convert_to_sympy_round(s)
    o2 = convert_to_sympy_round(s2)
    assert o.equals(o2)

def test_bracket_fix():
    s = '((((x_1+x_1)2)-(x_1-x_5))-(x_5/((x_0+0)/x_5)))'
    assert insert_mul_around_paren(s) == '((((x_1+x_1)*2)-(x_1-x_5))-(x_5/((x_0+0)/x_5)))'
