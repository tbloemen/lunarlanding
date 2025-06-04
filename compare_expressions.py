from sympy import *
from typing import List
import re
from functools import cache


def compare_multitrees(t1: List[str], t2: List[str]):
    """
    compares trees in the forest by comparing trees with same indices.
    If constants match when rounded, add 5 to similarity.
    If constants match when normalized (to 1 or -1 depending on the sign), add 1 to similarity
    """
    # for each kind of action
    assert len(t1) == len(t2)
    sim_score = 0
    for i in range(len(t1)):
        e_ones_1 = convert_to_sympy_ones(t1[i])
        e_ones_2 = convert_to_sympy_ones(t2[i])
        if e_ones_1.equals(e_ones_2):
            sim_score += 1
        else:
            e_round_1 = convert_to_sympy_round(t1[i])
            e_round_2 = convert_to_sympy_round(t2[i])
            sim_score += 4 if e_round_1.equals(e_round_2) else 0
    return sim_score


@cache
def convert_to_sympy_ones(exp: str) -> Expr:
    """
    converts string like this `((x_0/(x_3-(x_5+x_1)))/(((x_0*x_3)*x_3)*((x_4+x_0)/x_0)))` into a sympy expression.
    Each constant is replaced with -1 if it is negative, and with 1 if it is positive
    """
    exp = insert_mul_around_paren(exp)

    # Replace all numbers (not part of variable names)
    exp_clean = re.sub(r"(?<![a-zA-Z_])(\d+(\.\d+)?)", "1", exp)
    # Convert to sympy expression
    expr = sympify(exp_clean)
    return expr


@cache
def convert_to_sympy_round(exp: str) -> Expr:
    """
    converts string like this `((x_0/(x_3-(x_5+x_1)))/(((x_0*x_3)*x_3)*((x_4+x_0)/x_0)))` into a sympy expression.
    Each constant is replaced its rounded value
    """

    def repl_const_round(match):
        val = float(match.group(0))
        rounded = int(round(val))
        return "0.1" if rounded == 0 else str(rounded)

    exp = insert_mul_around_paren(exp)

    # Replace all numbers (not part of variable names)
    exp_clean = re.sub(r"(?<![a-zA-Z_])(\d+(\.\d+)?)", repl_const_round, exp)
    # Convert to sympy expression
    expr = sympify(exp_clean)
    return expr


def insert_mul_around_paren(exp: str) -> str:
    """
    Inserts a multiplication sign between:
    - a number or closing parenthesis and an opening parenthesis (e.g., '5(x+1)' or ')(' -> '5*(x+1)' or ')*(')
    - a closing parenthesis and a number (e.g., ')5' -> ')*5')
    """
    # Number or ')' before '('
    exp = re.sub(r"(\d+(\.\d+)?|\))\(", r"\1*(", exp)
    # ')' before number
    exp = re.sub(r"\)(\d+(\.\d+)?)", r")*\1", exp)
    return exp
