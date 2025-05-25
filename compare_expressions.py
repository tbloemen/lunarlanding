from sympy import *
from typing import List
import re

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
        e_round_1 = convert_ty_sympy(t1[i], repl_const_round)
        e_round_2 = convert_ty_sympy(t2[i], repl_const_round)
        if e_round_1.equals(e_round_2):
             sim_score += 5
        else:
            e_ones_1 = convert_ty_sympy(t1[i], repl_const_ones)
            e_ones_2 = convert_ty_sympy(t2[i], repl_const_ones)
            sim_score += 1 if e_ones_1.equals(e_ones_2) else 0
    return sim_score


def convert_ty_sympy(exp: str, replace_func) -> Expr:
    """
    converts string like this `((x_0/(x_3-(x_5+x_1)))/(((x_0*x_3)*x_3)*((x_4+x_0)/x_0)))` into a sympy expression.
    Each constant is replaced with -1 if it is negative, and with 1 if it is positive
    """
    # Replace negative numbers with -1, positive numbers with 1
    

    # Replace all numbers (not part of variable names) with -1 or 1
    exp_clean = re.sub(r'(?<![a-zA-Z_])(-?\d+(\.\d+)?)', replace_func, exp)

    # Convert to sympy expression
    expr = sympify(exp_clean)
    return expr


def repl_const_ones(match):
        val = float(match.group(0))
        return "-1" if val < 0 else "1"

def repl_const_round(match):
    val = float(match.group(0))
    return str(int(round(val)))